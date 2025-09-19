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
    ADDITIVE_UPDATE = "additive_update"  # Classic LoRA: W + AB^T
    LORA_PARA_COMPOSITION = "lorapara"  # (I-PP^T)W + AB^T


class KnowledgeInjectionLayer(nn.Module):
    """
    Knowledge injection that keeps original weights frozen while a small, trainable matrix
    injects new knowledge through principled low-rank updates.
    
    Supports four methods:
    1. Project-replace injection: Overwrite a sub-space of W
    2. Residual-projection injection: (I-P)W + PR (PaRa/Rank-reduction style)
    3. Additive update: Classic LoRA style W + AB^T
    4. LoRA+PaRa composition: (I-PP^T)W + AB^T (Independent LoRA and PaRa)
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
        
        if self.decomposition_method is None:
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
        
        # P matrix for projection (m×k) - Used by RESIDUAL_PROJECTION, PROJECT_REPLACE, and LORA_PARA_COMPOSITION
        if self.injection_method in [InjectionMethod.RESIDUAL_PROJECTION, 
                                   InjectionMethod.PROJECT_REPLACE, 
                                   InjectionMethod.LORA_PARA_COMPOSITION]:
            self.P = nn.Parameter(torch.zeros((self.out_features, self.r)))
        
        # For Project-Replace and Residual-Projection methods, we need R_new (k×n)
        if self.injection_method in [InjectionMethod.RESIDUAL_PROJECTION, InjectionMethod.PROJECT_REPLACE]:
            self.R_new = nn.Parameter(torch.zeros((self.r, self.in_features)))
        
        # For Additive update (LoRA) and LoRA+PaRa composition, we need A and B
        if self.injection_method in [InjectionMethod.ADDITIVE_UPDATE, InjectionMethod.LORA_PARA_COMPOSITION]:
            self.lora_A = nn.Parameter(torch.zeros((self.out_features, self.r)))
            self.lora_B = nn.Parameter(torch.zeros((self.r, self.in_features)))
        
        # Optional gating parameter
        if use_gating:
            self.gate = nn.Parameter(torch.ones(1) * 0.5)  # Initialize to 0.5
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize the parameters."""
        # Initialize P for QR stability (if it exists)
        if hasattr(self, 'P'):
            nn.init.normal_(self.P, mean=0.0, std=0.02)
        
        # Initialize R_new (if it exists)
        if hasattr(self, 'R_new'):
            nn.init.normal_(self.R_new, mean=0.0, std=0.02 * self.init_scale)
        
        # Initialize LoRA weights if needed
        if hasattr(self, 'lora_A') and hasattr(self, 'lora_B'):
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
    
    def count_parameters(self):
        """Count trainable parameters for this layer."""
        total_params = 0
        param_details = {}
        
        if hasattr(self, 'P'):
            p_params = self.P.numel()
            total_params += p_params
            param_details['P'] = p_params
            
        if hasattr(self, 'R_new'):
            r_params = self.R_new.numel()
            total_params += r_params
            param_details['R_new'] = r_params
            
        if hasattr(self, 'lora_A'):
            a_params = self.lora_A.numel()
            total_params += a_params
            param_details['lora_A'] = a_params
            
        if hasattr(self, 'lora_B'):
            b_params = self.lora_B.numel()
            total_params += b_params
            param_details['lora_B'] = b_params
            
        return total_params, param_details
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First apply the original layer to get the baseline output
        original_output = self.base_layer(x)
        
        # Apply dropout to input if needed
        x_dropped = self.lora_dropout(x)
        
        # Get original weights
        W = self.get_original_weights()
        
        if self.injection_method == InjectionMethod.PROJECT_REPLACE or \
           self.injection_method == InjectionMethod.RESIDUAL_PROJECTION:
            
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
            else:
                Q_P = self.P
            
            # Compute projection matrix
            P_proj = torch.matmul(Q_P, Q_P.transpose(0, 1))
            
            # Keep the part of W orthogonal to the projection space
            W_keep = W - torch.matmul(P_proj, W)
            
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
            
        elif self.injection_method == InjectionMethod.LORA_PARA_COMPOSITION:
            # LoRA+PaRa Composition: (I - PP^T)W + AB^T
            
            # Step 1: PaRa component - (I - PP^T)W
            if self.is_orthogonalize:
                if self.compute_svd_each_forward:
                    Q_P = self.orthogonalize(self.P)
                else:
                    if not hasattr(self, 'Q_P_cached') or self.Q_P_cached is None:
                        with torch.no_grad():
                            self.Q_P_cached = self.orthogonalize(self.P).detach()
                    Q_P = self.Q_P_cached
            else:
                Q_P = self.P
            
            # Compute projection matrix PP^T
            P_proj = torch.matmul(Q_P, Q_P.transpose(0, 1))
            
            # Apply PaRa: (I - PP^T)W
            W_para = W - torch.matmul(P_proj, W)
            
            # Apply PaRa transformation to input
            para_output = torch.matmul(x_dropped, W_para.transpose(0, 1))
            
            # Add bias if exists
            if hasattr(self.base_layer, 'bias') and self.base_layer.bias is not None:
                para_output += self.base_layer.bias
            
            # Step 2: LoRA component - AB^T
            lora_update = torch.matmul(
                torch.matmul(x_dropped, self.lora_B.transpose(0, 1)),
                self.lora_A.transpose(0, 1)
            ) * self.scaling
            
            # Apply optional gating
            if self.use_gating:
                lora_update = lora_update * torch.sigmoid(self.gate)
            
            # Step 3: Combine PaRa and LoRA
            return para_output + lora_update
        
        else:
            raise ValueError(f"Unsupported injection method: {self.injection_method}")


def count_model_parameters(model, method_name=""):
    """Helper function to count parameters by injection method."""
    total_params = 0
    method_breakdown = {}
    
    for name, module in model.named_modules():
        if isinstance(module, KnowledgeInjectionLayer):
            layer_params, param_details = module.count_parameters()
            total_params += layer_params
            method_breakdown[name] = param_details
    
    print(f"\n=== Parameter Count for {method_name} ===")
    print(f"Total trainable parameters: {total_params:,}")
    
    for layer_name, details in method_breakdown.items():
        print(f"{layer_name}: {sum(details.values()):,} params")
        for param_name, count in details.items():
            print(f"  {param_name}: {count:,}")
    
    return total_params