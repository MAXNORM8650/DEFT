# import packages
import math
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
class EffPaRaRankReductionLayer(nn.Module):
    """Optimized implementation of PaRaRankReduction technique."""
    
    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        truncated_svd: bool = False,
        update_freq: int = 10,  # Update projection matrix every N steps
        **kwargs,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.truncated_svd = truncated_svd
        self.update_freq = update_freq
        self.step_counter = 0
        
        # Determine base layer dimensions
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = base_layer.weight.shape
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        
        # Determine rank for decomposition
        self.r_out = min(r, out_features)
        
        # Create the learnable B matrix
        self.B = nn.Parameter(torch.zeros((out_features, self.r_out)))
        
        # Create a buffer for the cached projection matrix
        self.register_buffer('projection_matrix', torch.zeros((out_features, out_features)))
        self.projection_updated = False
        
        # Initialize weights
        if init_lora_weights:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
            
        # Update projection matrix once at initialization
        self._update_projection_matrix()
    
    def _update_projection_matrix(self):
        """Update the cached projection matrix based on current B values."""
        with torch.no_grad():
            if self.truncated_svd:
                # Using truncated SVD approach
                U, S, Vt = self._compute_tsvd()
                # projection = U × S × U^T
                self.projection_matrix.copy_(torch.matmul(U, torch.matmul(S, U.transpose(0, 1))))
            else:
                # Using QR decomposition approach
                B_float32 = self.B.to(torch.float32)
                Q, _ = torch.linalg.qr(B_float32)
                Q = Q.to(self.B.dtype)
                # projection = Q × Q^T
                self.projection_matrix.copy_(torch.matmul(Q, Q.transpose(0, 1)))
                
        self.projection_updated = True
    
    def _compute_tsvd(self):
        """Compute truncated SVD components for the B matrix."""
        # Cast to float32 for computation stability
        B = self.B.to(torch.float32)
        # Perform SVD
        U, S, Vt = torch.linalg.svd(B, full_matrices=False)
        # Cast back to original data type
        U = U.to(self.B.dtype)
        S = S.to(self.B.dtype)
        Vt = Vt.to(self.B.dtype)
        # Retain only the top 'r' singular values
        U = U[:, :self.r]
        S = torch.diag(S[:self.r])
        Vt = Vt[:self.r, :]
        
        return U, S, Vt
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Periodically update the projection matrix
        if self.training and (not self.projection_updated or self.step_counter % self.update_freq == 0):
            self._update_projection_matrix()
            
        if self.training:
            self.step_counter += 1
        
        # First apply the original layer
        original_output = self.base_layer(x)
        
        # Get the original weights
        if isinstance(self.base_layer, nn.Linear):
            original_weights = self.base_layer.weight
        elif isinstance(self.base_layer, nn.Conv2d):
            original_weights = self.base_layer.weight
        elif isinstance(self.base_layer, nn.Embedding):
            original_weights = self.base_layer.weight
        elif isinstance(self.base_layer, Conv1D):
            original_weights = self.base_layer.weight
        
        # Apply dropout to the input
        x_dropped = self.lora_dropout(x)
        
        # Compute the intermediate result: x_dropped @ W^T
        intermediate = torch.matmul(x_dropped, original_weights.transpose(0, 1))
        
        # Now apply the projection: (x_dropped @ W^T) @ projection_matrix
        adjustment = torch.matmul(intermediate, self.projection_matrix)
        
        # Compute the final output: original - adjustment
        return original_output - adjustment
    
    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.lora_alpha}, truncated_svd={self.truncated_svd}"