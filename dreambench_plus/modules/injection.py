import torch
import torch.nn as nn
import math
from enum import Enum
from typing import Optional, Union, Callable

class InjectionMethod(Enum):
    PROJECT_REPLACE = "project_replace"  # Overwrite a sub-space of W
    RESIDUAL_PROJECTION = "residual_projection"  # (I-P)W + PR
    ADDITIVE_UPDATE = "additive_update"  # Classic LoRA: W + AB^T

class KnowledgeInjectionLayer(nn.Module):
    """
    Knowledge injection that follows LoRA structure with two trainable matrices.
    Modified to avoid circular references causing recursion errors.
    """

    def init(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        injection_method: Union[str, InjectionMethod] = InjectionMethod.RESIDUAL_PROJECTION,
        network_alpha: Optional[float] = None,
        use_gating: bool = False,
        decomposition_method: str = "qr",
        dropout_rate: float = 0.0,
        init_scale: float = 1.0,
        original_module: Optional[nn.Module] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().init()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.network_alpha = network_alpha if network_alpha is not None else rank
        self.scaling = self.network_alpha / self.rank
        self.use_gating = use_gating
        self.decomposition_method = decomposition_method
        self.init_scale = init_scale

        # Store original weights separately to avoid circular references
        if original_module is not None:
            # Check if it's a ModuleList (like to_out in some attention modules)
            if isinstance(original_module, nn.ModuleList) and len(original_module) > 0:
                # Use the first element of the ModuleList which should be a Linear layer
                if hasattr(original_module[0], 'weight'):
                    self.register_buffer('original_weight', original_module[0].weight.detach().clone())
                    if hasattr(original_module[0], 'bias') and original_module[0].bias is not None:
                        self.register_buffer('original_bias', original_module[0].bias.detach().clone())
                    else:
                        self.register_buffer('original_bias', None)
                else:
                    raise AttributeError(f"Module {type(original_module[0])} does not have a weight attribute")
            # Regular Linear layer
            elif hasattr(original_module, 'weight'):
                self.register_buffer('original_weight', original_module.weight.detach().clone())
                if hasattr(original_module, 'bias') and original_module.bias is not None:
                    self.register_buffer('original_bias', original_module.bias.detach().clone())
                else:
                    self.register_buffer('original_bias', None)
            else:
                raise AttributeError(f"Module {type(original_module)} does not have a weight attribute")
        else:
            self.register_buffer('original_weight', None)
            self.register_buffer('original_bias', None)

        # Convert string to enum if necessary
        if isinstance(injection_method, str):
            try:
                self.injection_method = InjectionMethod(injection_method.lower())
            except ValueError:
                raise ValueError(f"Unknown injection method: {injection_method}. "
                               f"Available methods: {[m.value for m in InjectionMethod]}")
        else:
            self.injection_method = injection_method

        # Optional dropout
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

        # Following LoRA structure with down and up projection
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)

        # For orthogonal projection methods, we need a projection matrix
        if self.injection_method in [InjectionMethod.PROJECT_REPLACE, InjectionMethod.RESIDUAL_PROJECTION]:
            # Register buffer for caching the orthogonal projection
            self.register_buffer('orthogonal_proj', None, persistent=False)

        # Optional gating parameter
        if use_gating:
            self.gate = nn.Parameter(torch.ones(1) * 0.5)  # Initialize to 0.5

        # Initialize weights
        self.reset_parameters()

    def repr(self):
        """Custom representation to avoid recursion error during printing"""
        return f"KnowledgeInjectionLayer(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, method={self.injection_method.value})"

    def reset_parameters(self):
        """Initialize the parameters similar to LoRA."""
        if self.injection_method == InjectionMethod.ADDITIVEUPDATE:
            # Standard LoRA initialization
            nn.init.normal(self.down.weight, std=1 / self.rank)
            nn.init.zeros(self.up.weight)
        else:
            # For projection methods, different initialization might help
            nn.init.normal(self.down.weight, std=0.02)
            nn.init.normal_(self.up.weight, std=0.02 * self.init_scale)

    def qr_orthogonalize(self, matrix):
        """QR-orthogonalize a matrix to get Q."""
        # Ensure numerical stability by using float32
        matrixfloat = matrix.to(torch.float32)
        Q,  = torch.linalg.qr(matrix_float)
        # Convert back to original dtype
        return Q.to(matrix.dtype)

    def svd_orthogonalize(self, matrix):
        """SVD-orthogonalize a matrix to get U."""
        # Ensure numerical stability by using float32
        matrixfloat = matrix.to(torch.float32)
        U, , _ = torch.linalg.svd(matrix_float, full_matrices=False)
        # Take only the first r columns
        U = U[:, :self.rank]
        # Convert back to original dtype
        return U.to(matrix.dtype)

    def get_orthogonal_matrix(self):
        """Get orthogonal matrix from up matrix weights."""
        if self.decomposition_method == "qr":
            return self.qr_orthogonalize(self.up.weight.T)
        elif self.decomposition_method == "svd":
            return self.svd_orthogonalize(self.up.weight.T)
        else:
            raise ValueError(f"Unknown decomposition method: {self.decomposition_method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        dtype = self.down.weight.dtype
        x = x.to(dtype)

        # Apply dropout to input if needed
        x_dropped = self.dropout(x)

        # Calculate the down projection (similar to LoRA)
        down_hidden_states = self.down(x_dropped)

        if self.injection_method == InjectionMethod.ADDITIVE_UPDATE:
            # Standard LoRA approach: W + AB^T
            up_hidden_states = self.up(down_hidden_states)

            # Apply scaling
            if self.network_alpha is not None:
                up_hidden_states = up_hidden_states * self.scaling

            # Apply optional gating
            if self.use_gating:
                up_hidden_states = up_hidden_states * torch.sigmoid(self.gate)

            return up_hidden_states.to(orig_dtype)

        elif self.injection_method in [InjectionMethod.PROJECT_REPLACE, InjectionMethod.RESIDUAL_PROJECTION]:
            # We need the original weights for these methods
            if self.original_weight is None:
                raise ValueError("Original weights must be provided for PROJECT_REPLACE or RESIDUAL_PROJECTION methods")

            W = self.original_weight

            # Get orthogonal projection matrix if not cached or in training mode
            if self.orthogonal_proj is None or self.training:
                Q_P = self.get_orthogonal_matrix()
                # Cache for inference
                if not self.training:
                    self.orthogonal_proj = Q_P
            else:
                Q_P = self.orthogonal_proj

            # Create projection matrix Q_P * Q_P^T
            P_proj = torch.matmul(Q_P, Q_P.transpose(0, 1))

            # W_keep = (I - Q_PQ_P^T)W - the part of W we keep
            W_keep = W - torch.matmul(P_proj, W)

            # Inject new content: Q_P * (result from down projection)
            R_new = self.down.weight
            W_inject = torch.matmul(Q_P, R_new)

            # Apply optional gating
            if self.use_gating:
                W_inject = W_inject * torch.sigmoid(self.gate)

            # Final result using matrix multiplication approach
            result = torch.matmul(x_dropped, (W_keep + W_inject).transpose(0, 1))

            # Add bias if available
            if self.original_bias is not None:
                result += self.original_bias

            return result.to(orig_dtype)

        else:
            raise ValueError(f"Unsupported injection method: {self.injection_method}")
# Add "set lora layer" pattern like in diffusers
def inject_knowledge_layer(
    module: nn.Module,
    rank: int = 8,
    injection_method: Union[str, InjectionMethod] = InjectionMethod.PROJECT_REPLACE,
    network_alpha: Optional[float] = None,
    use_gating: bool = False,
    decomposition_method: str = "qr",
    dropout_rate: float = 0.0,
    init_scale: float = 1.0,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Utility function to replace module in-place with a KnowledgeInjectionLayer.
    Similar to applying LoRA to a module.
    """
    if not isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
        raise ValueError(f"Module {type(module)} not supported for knowledge injection")

    if isinstance(module, nn.Linear):
        in_features, out_features = module.in_features, module.out_features
        module.lora_layer = KnowledgeInjectionLayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            injection_method=injection_method,
            network_alpha=network_alpha,
            use_gating=use_gating,
            decomposition_method=decomposition_method,
            dropout_rate=dropout_rate,
            init_scale=init_scale,
            original_module=module,
            device=device,
            dtype=dtype
        )

        # Save the original forward method
        module._original_forward = module.forward

        # Replace the forward method
        def wrapped_forward(self, x):
            if self.lora_layer.injection_method == InjectionMethod.ADDITIVE_UPDATE:
                return self._original_forward(x) + self.lora_layer(x)
            else:
                return self.lora_layer(x, self)

        # Bind the new forward method to the module
        import types
        module.forward = types.MethodType(wrapped_forward, module)

    # Similar implementation for Conv2d and Embedding would go here

    return module

# Monkey patch nn.Linear with a set_lora_layer method to match diffusers API
def set_knowledge_injection_layer(self, lora_layer):
    """
    Method to add to Linear layers to set a knowledge injection layer.
    This is similar to LoRA's approach but for knowledge injection.
    """
    self.lora_layer = lora_layer
    
    # Save the original forward method if not already saved
    if not hasattr(self, '_original_forward'):
        self._original_forward = self.forward
    
    # Replace the forward method to include knowledge injection
    # Modified to handle variable arguments
    def wrapped_forward(self, x, *args, **kwargs):
        if self.lora_layer.injection_method == InjectionMethod.ADDITIVE_UPDATE:
            return self._original_forward(x, *args, **kwargs) + self.lora_layer(x)
        else:
            return self.lora_layer(x, self)
    
    # Bind the new forward method to the module
    import types
    self.forward = types.MethodType(wrapped_forward, self)

# Add the method to nn.Linear
nn.Linear.set_knowledge_injection_layer = set_knowledge_injection_layer