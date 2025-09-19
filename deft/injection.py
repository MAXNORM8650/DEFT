import torch
import torch.nn as nn
import math
from enum import Enum
from typing import Optional, Union, Callable
import logging
from pathlib import Path


class InjectionMethod(Enum):
    PROJECT_REPLACE = "project_replace"  # Overwrite a sub-space of W
    RESIDUAL_PROJECTION = "residual_projection"  # (I-P)W + PR
    ADDITIVE_UPDATE = "additive_update"  # LoRA style : W + AB^T

class KnowledgeInjectionLayer(nn.Module):
    """
    Knowledge injection that keeps original weights frozen while a small, trainable matrix
    injects new knowledge through principled low-rank updates.
    
    Supports three methods:
    1. Project-replace injection: Overwrite a sub-space of W
    2. Residual-projection injection: (I-P)W + PR (PaRa/Rank-reduction style)
    3. Additive update: Classic LoRA style W + AB^T
    """
    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        injection_method: Union[str, InjectionMethod] = InjectionMethod.RESIDUAL_PROJECTION,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        init_scale: float = 0.25,
        use_gating: bool = False,
        decomposition_method: str = "qr",
        compute_svd_each_forward: bool = True,
        orthogonalize: bool = False,
        **kwargs,
    ):
        super().__init__()
        # orthogonalize = False
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.use_gating = use_gating
        self.init_scale = init_scale
        self.compute_svd_each_forward = compute_svd_each_forward
        self.decomposition_method = decomposition_method
        # breakpoint()
        if self.decomposition_method =='None':
            #logging.warning("Decomposition method is None, defaulting to 'qr'")
            logging.warning("Decomposition method is None, defaulting to non-orthogonalized")
            self.is_orthogonalize = False
        else:
            self.is_orthogonalize = True
            logging.info(f"Using decomposition method: {self.decomposition_method}")
        # Convert string to enum if necessary
        if isinstance(injection_method, str):
            try:
                self.injection_method = InjectionMethod(injection_method.lower())
            except ValueError:
                raise ValueError(f"Unknown injection method: {injection_method}. "
                                f"Available methods: {[m.value for m in InjectionMethod]}")
        else:
            self.injection_method = injection_method
        
        # Determine base layer dimensions
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            self.in_features, self.out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            self.in_features, self.out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif hasattr(base_layer, 'weight') and hasattr(base_layer.weight, 'shape'):
            if len(base_layer.weight.shape) == 2:
                self.out_features, self.in_features = base_layer.weight.shape
            else:
                raise ValueError(f"Unsupported weight shape {base_layer.weight.shape}")
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        self.register_buffer('Q_P_cached', None)
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        
        # P matrix for projection (m×k)
        self.P = nn.Parameter(torch.zeros((self.out_features, self.r)))
        
        # For Project-Replace and Residual-Projection methods, we need R_new (k×n)
        self.R_new = nn.Parameter(torch.zeros((self.r, self.in_features)))
        
        # For Additive update (LoRA), we need A and B
        if self.injection_method == InjectionMethod.ADDITIVE_UPDATE:
            self.lora_A = nn.Parameter(torch.zeros((self.out_features, self.r)))
            self.lora_B = nn.Parameter(torch.zeros((self.r, self.in_features)))
        
        # Optional gating parameter
        if use_gating:
            self.gate = nn.Parameter(torch.ones(1) * 0.5)  # Initialize to 0.5
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize the parameters."""
        # Initialize P for QR stability
        nn.init.normal_(self.P, mean=0.0, std=0.02)
        
        # Initialize R_new
        if self.injection_method != InjectionMethod.ADDITIVE_UPDATE:
            nn.init.normal_(self.R_new, mean=0.0, std=0.02 * self.init_scale)
        
        # Initialize LoRA weights if needed
        if self.injection_method == InjectionMethod.ADDITIVE_UPDATE:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def qr_orthogonalize(self, P):
        """QR-orthogonalize the P matrix to get Q_P."""
        # Ensure numerical stability by using float32
        P_float = P.to(torch.float32)
        Q, _ = torch.linalg.qr(P_float)
        # Convert back to original dtype
        return Q.to(P.dtype)

    def svd_orthogonalize(self, P):
        """SVD-orthogonalize the P matrix to get Q_P."""
        # Ensure numerical stability by using float32
        P_float = P.to(torch.float32)
        U, _, _ = torch.linalg.svd(P_float, full_matrices=False)
        # Take only the first r columns
        U = U[:, :self.r]
        # Convert back to original dtype
        return U.to(P.dtype)
    
    def orthogonalize(self, P):
        """Orthogonalize P matrix based on specified method."""
        if self.decomposition_method == "qr":
            return self.qr_orthogonalize(P)
        elif self.decomposition_method == "svd":
            return self.svd_orthogonalize(P)
        else:
            raise ValueError(f"Unknown decomposition method: {self.decomposition_method}")
    
    def get_original_weights(self):
        """Helper method to get original weights from base layer."""
        if hasattr(self.base_layer, 'weight'):
            return self.base_layer.weight
        raise AttributeError(f"Base layer {type(self.base_layer)} does not have a weight attribute")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First apply the original layer to get the baseline output
        original_output = self.base_layer(x)
        
        # Apply dropout to input if needed
        x_dropped = self.lora_dropout(x)
        
        # Get original weights
        W = self.get_original_weights()
        
        # Create local Q_P without modifying self attributes during forward
        if self.is_orthogonalize:
            if self.compute_svd_each_forward:
                # Compute on the fly without caching
                Q_P = self.orthogonalize(self.P)
            else:
                # Use cached version or compute and cache
                if not hasattr(self, 'Q_P_cached') or self.Q_P_cached is None:
                    with torch.no_grad():  # Use no_grad for caching to avoid extra graph connections
                        self.Q_P_cached = self.orthogonalize(self.P).detach()
                Q_P = self.Q_P_cached
            P_proj = torch.matmul(Q_P, Q_P.transpose(0, 1))
        else:
            Q_P = self.P
            # P_proj = torch.relu(torch.matmul(Q_P, Q_P.transpose(0, 1))) # High control in new concepts but less quality (High clip score)
            P_proj = torch.matmul(Q_P, torch.relu(Q_P).transpose(0, 1))
            # P_proj = torch.matmul(torch.relu(Q_P), torch.relu(Q_P).transpose(0, 1)) # High control with relatively good quality and propmts (High clip score)
            # P_proj = torch.matmul(Q_P, Q_P.transpose(0, 1)) # High quality as well as good control but lack with clip scores. (High clip score)
        
        # Compute projection matrix
        # P_proj = torch.matmul(torch.relu(Q_P), torch.relu(Q_P).transpose(0, 1))
        # Continue with your method-specific implementations
        if self.injection_method == InjectionMethod.PROJECT_REPLACE or \
        self.injection_method == InjectionMethod.RESIDUAL_PROJECTION:
            # Keep the part of W orthogonal to the projection space
            W_keep = W - torch.matmul(P_proj, W)
            # torch.relu(self.R_new)
            # Inject new content in the projection space
            W_inject = torch.matmul(Q_P, self.R_new)
            
            # Apply optional gating
            if self.use_gating:
                W_inject = W_inject * torch.sigmoid(self.gate)
            
            # Get the modified weight
            W_modified = W_keep + W_inject
            
            # Apply the modified weight with the input
            updated_output = torch.matmul(x_dropped, W_modified.transpose(0, 1))
            
            if hasattr(self.base_layer, 'bias') and self.base_layer.bias is not None:
                updated_output += self.base_layer.bias
            
            return updated_output
            
        elif self.injection_method == InjectionMethod.ADDITIVE_UPDATE:
            # Compute the LoRA update
            lora_update = torch.matmul(
                torch.matmul(x_dropped, self.lora_B.transpose(0, 1)),
                self.lora_A.transpose(0, 1)
            ) * self.scaling
            
            # Apply optional gating
            if self.use_gating:
                lora_update = lora_update * torch.sigmoid(self.gate)
            
            # Add the LoRA update to the original output
            return original_output + lora_update
        
        else:
            raise ValueError(f"Unsupported injection method: {self.injection_method}")


class KnowledgeInjectionAdapter(nn.Module):
    """
    Adapter that applies knowledge injection to specific layers in a model.
    This is a convenience wrapper for applying KnowledgeInjectionLayer to multiple layers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        r: int = 8,
        injection_method: Union[str, InjectionMethod] = InjectionMethod.PROJECT_REPLACE,
        target_modules: list = None,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        init_scale: float = 1.0,
        use_gating: bool = False,
        decomposition_method: str = "qr",
        compute_svd_each_forward: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.injection_layers = nn.ModuleDict()
        
        if target_modules is None:
            # Default to applying to all Linear layers
            target_modules = [name for name, module in model.named_modules() 
                             if isinstance(module, nn.Linear)]
        
        # Apply knowledge injection to target modules
        for name, module in model.named_modules():
            if name in target_modules:
                self.injection_layers[name] = KnowledgeInjectionLayer(
                    base_layer=module,
                    r=r,
                    injection_method=injection_method,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    init_scale=init_scale,
                    use_gating=use_gating,
                    decomposition_method=decomposition_method,
                    compute_svd_each_forward=compute_svd_each_forward,
                    **kwargs,
                )
                
                # Replace the original module with the injection layer
                self._replace_module(model, name, self.injection_layers[name])
    
    def _replace_module(self, root_module, target_name, replacement_module):
        """Replace a module in the model hierarchy by its name."""
        parent_name = '.'.join(target_name.split('.')[:-1])
        child_name = target_name.split('.')[-1]
        
        if parent_name:
            parent_module = root_module
            for part in parent_name.split('.'):
                parent_module = getattr(parent_module, part)
            setattr(parent_module, child_name, replacement_module)
        else:
            setattr(root_module, child_name, replacement_module)
    
    def forward(self, *args, **kwargs):
        """Forward pass is just passing through to the adapted model."""
        return self.model(*args, **kwargs)
    
    def get_trainable_parameters(self):
        """Get all trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]


if __name__ == "__main__":
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(768, 768)
            self.linear2 = nn.Linear(768, 768)
            self.out = nn.Linear(768, 10)
        
        def forward(self, x):
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            x = torch.relu(x)
            return self.out(x)
    
    # Create model
    model = SimpleModel()
    
    # Freeze original parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply knowledge injection
    adapted_model = KnowledgeInjectionAdapter(
        model=model,
        r=16,
        injection_method="project_replace",  # Can be "project_replace", "residual_projection", or "additive_update"
        target_modules=["linear1", "linear2"],
        lora_alpha=16,
        lora_dropout=0.1,
        use_gating=False,
        decomposition_method="qr",
    )
    
    # Generate a sample input
    x = torch.randn(32, 768)
    
    # Forward pass
    output = adapted_model(x)
    print(f"Output shape: {output.shape}")  # Should be [32, 10]
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in adapted_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    print(f"Parameter efficiency: {trainable_params / total_params:.2%}")