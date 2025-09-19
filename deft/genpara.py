import torch
import torch.nn as nn
import math
import numpy as np
from enum import Enum
from typing import Optional, Union, Callable


class DecompositionMethod(Enum):
    QR = "qr"
    TSVD = "tsvd"
    LRMF = "lrmf"
    NMF = "nmf"
    EIGEN = "eigen"
    # LORAPARA = "lorapara"
class GeneralizedPaRaRankReductionLayer(nn.Module):
    """
    Generalized implementation of PaRaRankReduction technique with multiple decomposition methods.
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        decomposition_method: Union[str, DecompositionMethod] = DecompositionMethod.QR,
        adaptive_rank: bool = False,
        variance_threshold: float = 0.95,
        sparse_parameterization: bool = False,
        sparsity_factor: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.base_layer = base_layer
        self.r = r  # Initial rank
        self.lora_alpha = lora_alpha
        self.adaptive_rank = adaptive_rank
        self.variance_threshold = variance_threshold
        self.sparse_parameterization = sparse_parameterization
        self.sparsity_factor = sparsity_factor
        
        # Convert string to enum if necessary
        if isinstance(decomposition_method, str):
            try:
                self.decomposition_method = DecompositionMethod(decomposition_method.lower())
            except ValueError:
                raise ValueError(f"Unknown decomposition method: {decomposition_method}. "
                                f"Available methods: {[m.value for m in DecompositionMethod]}")
        else:
            self.decomposition_method = decomposition_method
        
        # Determine base layer dimensions
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif hasattr(base_layer, 'weight') and hasattr(base_layer.weight, 'shape'):
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
        
        # Create the learnable B matrix
        if sparse_parameterization:
            # For sparse parameterization, we create a mask and a dense matrix
            self.register_buffer(
                "sparse_mask", 
                torch.bernoulli(torch.ones(out_features, r) * (1 - sparsity_factor))
            )
            self.B_dense = nn.Parameter(torch.zeros((out_features, r)))
            # Initialize weights
            if init_lora_weights:
                nn.init.kaiming_uniform_(self.B_dense, a=math.sqrt(5))
        else:
            # Standard dense parameterization
            self.B = nn.Parameter(torch.zeros((out_features, r)))
            # Initialize weights
            if init_lora_weights:
                nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        
        # For NMF, ensure parameters are non-negative
        if self.decomposition_method == DecompositionMethod.NMF:
            self.W = nn.Parameter(torch.abs(torch.randn((out_features, r))))
            self.H = nn.Parameter(torch.abs(torch.randn((r, out_features))))
    
    @property
    def B_effective(self):
        """Get the effective B matrix, which may be sparse if sparse_parameterization is True."""
        if self.sparse_parameterization:
            return self.B_dense * self.sparse_mask
        return self.B
    
    def get_rank(self, S=None):
        """
        Determine the rank to use based on adaptive settings or return the fixed rank.
        
        Args:
            S: Optional singular values if already computed
            
        Returns:
            int: The rank to use
        """
        if not self.adaptive_rank:
            return self.r
        
        if S is None:
            # If S is not provided, compute SVD
            _, S, _ = torch.linalg.svd(self.B_effective.to(torch.float32), full_matrices=False)
        
        # Compute cumulative explained variance
        cumulative_variance = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
        
        # Find the smallest k such that we retain variance_threshold of variance
        k = torch.nonzero(cumulative_variance >= self.variance_threshold, as_tuple=True)[0]
        if len(k) > 0:
            return k[0].item() + 1  # +1 because we want to include this component
        return self.r  # Fallback to default rank

    
    def qr_decomposition(self):
        """Perform QR decomposition on B matrix."""
        B_float32 = self.B_effective.to(torch.float32)
        Q, _ = torch.linalg.qr(B_float32)
        Q = Q.to(self.B_effective.dtype)
        return Q
    
    def tsvd(self):
        """Perform truncated SVD on B matrix."""
        B = self.B_effective.to(torch.float32)
        U, S, Vt = torch.linalg.svd(B, full_matrices=False)
        
        if self.adaptive_rank:
            r = self.get_rank(S)
        else:
            r = min(self.r, self.out_features)
        
        U = U[:, :r].to(self.B_effective.dtype)
        S = torch.diag(S[:r]).to(self.B_effective.dtype)
        Vt = Vt[:r, :].to(self.B_effective.dtype)
        
        return U, S, Vt
    
    def lrmf(self):
        """
        Perform Low-Rank Matrix Factorization that returns factors suitable for creating
        a projection matrix with the correct dimensions.
        """
        B = self.B_effective.to(torch.float32)
        U, S, Vt = torch.linalg.svd(B, full_matrices=False)
        
        if self.adaptive_rank:
            r = self.get_rank(S)
        else:
            r = min(self.r, self.out_features)
        
        # For LRMF, we need to ensure the projection matrix will have shape [out_features, out_features]
        # We're only using U here as we need a square projection matrix
        U = U[:, :r].to(self.B_effective.dtype)
        S_sqrt = torch.sqrt(S[:r]).to(self.B_effective.dtype)
        
        return U, S_sqrt

    
    def nmf(self):
        """Non-Negative Matrix Factorization - return W and H factors."""
        # Ensure non-negative parameters
        W_positive = torch.relu(self.W)
        H_positive = torch.relu(self.H)
        
        return W_positive, H_positive
    
    def eigendecomposition(self):
        """
        Robust eigenvalue decomposition of B*B^T that handles numerical instability.
        """
        B = self.B_effective.to(torch.float32)
        
        # Form symmetric matrix B*B^T for eigendecomposition
        BBT = torch.matmul(B, B.t())
        
        # Add a small regularization term to improve conditioning
        reg_term = torch.eye(BBT.shape[0], device=BBT.device) * 1e-6
        BBT_reg = BBT + reg_term
        
        try:
            # Try with the regularized matrix
            # "linalg_eigh_cuda" not implemented for 'BFloat16', so convert to float32
            BBT_reg = BBT_reg.to(torch.float32)
            eigenvalues, eigenvectors = torch.linalg.eigh(BBT_reg)
            eigenvalues = eigenvalues.to(self.B_effective.dtype)
            eigenvectors = eigenvectors.to(self.B_effective.dtype)
        except Exception as e:
            # If that fails, fall back to SVD which is more numerically stable
            print(f"Warning: Eigendecomposition failed ({str(e)}). Falling back to SVD.")

            U, S, _ = torch.linalg.svd(B, full_matrices=False)
            eigenvalues = S**2  # Squared singular values are eigenvalues of B*B^T
            eigenvalues = eigenvalues.to(self.B_effective.dtype)
            eigenvectors = U
            eigenvectors = eigenvectors.to(self.B_effective.dtype)
        
        # Sort eigenvalues in descending order
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Filter out extremely small negative eigenvalues that might occur due to numerical issues
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        
        if self.adaptive_rank:
            # Only consider eigenvalues that are significant
            significant_eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(significant_eigenvalues) == 0:
                r = 1  # At least use 1 component
            else:
                total_variance = torch.sum(significant_eigenvalues)
                cumulative_variance = torch.cumsum(significant_eigenvalues, dim=0) / total_variance
                
                # Find the rank that captures the desired variance
                indices = torch.nonzero(cumulative_variance >= self.variance_threshold, as_tuple=True)[0]
                r = indices[0].item() + 1 if len(indices) > 0 else 1
                r = min(r, self.r, self.out_features)
        else:
            r = min(self.r, self.out_features)
        
        # Retain only the top 'r' eigenvectors and values
        eigenvectors = eigenvectors[:, :r].to(self.B_effective.dtype)
        eigenvalues = eigenvalues[:r].to(self.B_effective.dtype)
        
        return eigenvectors, torch.diag(eigenvalues)
    
    def _eigendecomposition(self):
        """Eigenvalue decomposition of B*B^T."""
        B = self.B_effective.to(torch.float32)
        # Form symmetric matrix B*B^T for eigendecomposition
        BBT = torch.matmul(B, B.t())
        # "linalg_eigh_cuda" not implemented for 'BFloat16', so convert to float32
        BBT = BBT.to(torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(BBT)
        # Convert back to original dtype
        eigenvalues = eigenvalues.to(self.B_effective.dtype)
        # Sort eigenvalues in descending order
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        if self.adaptive_rank:
            total_variance = torch.sum(eigenvalues)
            cumulative_variance = torch.cumsum(eigenvalues, dim=0) / total_variance
            r = torch.nonzero(cumulative_variance >= self.variance_threshold, as_tuple=True)[0][0].item() + 1
            r = min(r, self.r)
        else:
            r = min(self.r, self.out_features)
        
        # Retain only the top 'r' eigenvectors and values
        eigenvectors = eigenvectors[:, :r].to(self.B_effective.dtype)
        eigenvalues = eigenvalues[:r].to(self.B_effective.dtype)
        
        return eigenvectors, torch.diag(eigenvalues)
    
    def compute_projection_matrix(self):
        """
        Compute the projection matrix based on the selected decomposition method.
        
        Returns:
            torch.Tensor: The projection matrix
        """
        if self.decomposition_method == DecompositionMethod.QR:
            Q = self.qr_decomposition()
            return torch.matmul(Q, Q.transpose(0, 1))
        
        elif self.decomposition_method == DecompositionMethod.TSVD:
            U, S, Vt = self.tsvd()
            return torch.matmul(U, torch.matmul(S, U.transpose(0, 1)))
        
        elif self.decomposition_method == DecompositionMethod.LRMF:
            U, S_sqrt = self.lrmf()
            # Create projection matrix as U * U.T with scaling from singular values
            weighted_U = U * S_sqrt.unsqueeze(0)
            return torch.matmul(weighted_U, weighted_U.transpose(0, 1))
        
        elif self.decomposition_method == DecompositionMethod.NMF:
            W, H = self.nmf()
            return torch.matmul(W, H)
        
        elif self.decomposition_method == DecompositionMethod.EIGEN:
            eigenvectors, eigenvalues = self.eigendecomposition()
            return torch.matmul(eigenvectors, torch.matmul(eigenvalues, eigenvectors.transpose(0, 1)))
        
        else:
            raise ValueError(f"Unsupported decomposition method: {self.decomposition_method}")
    
    def get_original_weights(self):
        """Helper method to get original weights from base layer."""
        if hasattr(self.base_layer, 'weight'):
            return self.base_layer.weight
        raise AttributeError(f"Base layer {type(self.base_layer)} does not have a weight attribute")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First apply the original layer
        original_output = self.base_layer(x)
        
        # Get the original weights
        original_weights = self.get_original_weights()
        
        # Apply dropout to the input
        x_dropped = self.lora_dropout(x)
        
        # Compute the projection matrix
        projection_matrix = self.compute_projection_matrix()
        
        # Apply the adjustment: original - (x @ W^T @ projection)
        # breakpoint()
        adjustment = torch.matmul(x_dropped, original_weights.transpose(0, 1))
        adjustment = torch.matmul(adjustment, projection_matrix)
        
        # Return original output with adjustment
        return original_output - adjustment


# Example usage
if __name__ == "__main__":
    # Create a base linear layer
    base_layer = nn.Linear(768, 768)
    
    # Create a generalized PaRaRank reduction layer
    para_layer = GeneralizedPaRaRankReductionLayer(
        base_layer=base_layer,
        r=16,
        lora_alpha=8,
        lora_dropout=0.1,
        decomposition_method="eigen",
        adaptive_rank=True,
        variance_threshold=0.95,
        sparse_parameterization=True,
        sparsity_factor=0.2
    )
    
    # Generate a sample input
    x = torch.randn(32, 768)
    
    # Forward pass
    output = para_layer(x)
    print(f"Output shape: {output.shape}")  # Should be [32, 768]