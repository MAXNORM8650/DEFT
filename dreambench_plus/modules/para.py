import math
import types
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.lora import LoraConfig
from .genpara import GeneralizedPaRaRankReductionLayer

class PaRaRankReductionLayer(nn.Module):
    """Implementation of PaRaRankReduction technique."""
    
    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        truncated_svd: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        
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
        
        #if using truncated SVD if transcated_svd flag is True else same
        # breakpoint()
        if truncated_svd:
            self.truncated_svd = True
            # Full matrix transformation 
            r_out = min(r, out_features)
            # print("Using truncated SVD for PaRaRankReduction")

        else:
            self.truncated_svd = False
            # print("Not using truncated SVD for PaRaRankReduction")
        # Create the learnable B matrix (for QR decomposition)
        self.B = nn.Parameter(torch.zeros((out_features, r)))
        # Initialize weights
        if init_lora_weights:
            # Initialize B with a small random value
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
        # Apply the lora_dropout to the input
        x_dropped = self.lora_dropout(x)
        if self.truncated_svd:
            U, S, Vt = self.tsvd()
            
            # First compute U × S × Vt to get the projection matrix (out_features × out_features)
            # Note: Vt is (r, r), but we need a square matrix for the projection
            # So the correct computation should be U × S × U^T
            projection = torch.matmul(U, torch.matmul(S, U.transpose(0, 1)))
            
            # Now compute the adjustment
            adjustment = torch.matmul(x_dropped, original_weights.transpose(0, 1))
            adjustment = torch.matmul(adjustment, projection)
            
            return original_output - adjustment
        # Cast to float32 for QR decomposition (which doesn't support BFloat16)
        B_float32 = self.B.to(torch.float32)
        
        # Apply QR decomposition to get orthogonal matrix Q
        Q, _ = torch.linalg.qr(B_float32)  # Q ∈ R^(out_features × r)
        
        # Cast Q back to the original data type of B
        Q = Q.to(self.B.dtype)
        
        # Compute Q * Q^T
        QQT = torch.matmul(Q, Q.transpose(0, 1))  # Q * Q^T ∈ R^(out_features × out_features)
        
        # Apply the PaRa adjustment: W - Q*Q^T*W
        # For efficiency, we'll compute directly with the input
        
        # Get the adjustment part: (Q*Q^T*W) @ x
        adjustment = torch.matmul(x_dropped, original_weights.transpose(0, 1))
        adjustment = torch.matmul(adjustment, QQT)
        
        # Compute the final output: original - adjustment
        return original_output - adjustment

    def tsvd(self):
        """
        Perform truncated SVD for low-rank approximation of A
        Arguments:
        A: Input matrix
        rank: The number of singular values to retain

        Returns:
        U: Left singular vectors
        S: Singular values (diagonal matrix)
        Vt: Right singular vectors (transposed)
        """
         # Cast to float32 for QR decomposition (which doesn't support BFloat16)
        B = self.B.to(torch.float32)
        # Perform SVD
        U, S, Vt = torch.linalg.svd(B, full_matrices=False)
        # Cast back to the original data type
        U = U.to(self.B.dtype)
        S = S.to(self.B.dtype)
        Vt = Vt.to(self.B.dtype)
        # Retain only the top 'rank' singular values
        U = U[:, :self.r]
        S = torch.diag(S[:self.r])
        Vt = Vt[:self.r, :]
        
        return U, S, Vt


def make_para_rank_adapter(model, config):
    """
    Apply PaRaRankReduction to a model by directly modifying it rather than wrapping it.
    
    Args:
        model: The model to modify
        config: The LoraConfig containing the PaRaRankReduction parameters
        
    Returns:
        The modified model with PaRaRankReduction applied
    """
    # Track modified modules to avoid double-application
    modified_modules = set()
    
    # Recursive function to find and replace target modules
    def replace_modules(module, parent_name=""):
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            # Skip if already modified
            if full_name in modified_modules:
                continue
            if config.target_modules is None:
                config.target_modules = []

            if config.target_modules is None or len(config.target_modules) == 0:
                # Automatically identify the target modules (e.g., Linear layers)
                is_target = isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d)  # Customize this line for your case
            else:
                # Check if this module is part of the target_modules list
                is_target = any(target_key in full_name for target_key in config.target_modules)

            if is_target and isinstance(child, nn.Linear):
                config_dict = config.to_dict()
                # config_dict["decomposition_method"] = "lrmf"
                # breakpoint()
                para_layer = GeneralizedPaRaRankReductionLayer(base_layer=child, **config_dict)
                # para_layer = PaRaRankReductionLayer(
                #     base_layer=child,
                #     r=config.r,
                #     lora_alpha=config.lora_alpha,
                #     lora_dropout=config.lora_dropout,
                #     init_lora_weights=config.init_lora_weights,
                # )
                setattr(module, name, para_layer)
                modified_modules.add(full_name)
            else:
                # Recursively search for target modules
                replace_modules(child, full_name)
    
    # If model has an llm attribute, apply to that specifically
    if hasattr(model, "llm"):
        replace_modules(model.llm, "llm")
    else:
        replace_modules(model)
    
    # Add utility methods to the model
    def get_trainable_parameters(self):
        """Get parameters that require gradients for training."""
        return [p for n, p in self.named_parameters() if "B" in n and p.requires_grad]
    
    model.get_trainable_parameters = types.MethodType(get_trainable_parameters, model)
    
    # Add a marker to indicate PaRaRankReduction has been applied
    model._para_rank_applied = True
    model._para_config = config
    
    print(f"Applied Gen PaRaRankReduction to {len(modified_modules)} modules")
    
    return model


def apply_para_rank_to_model(model, target_modules=None, rank=8, lora_alpha=8, lora_dropout=0.1, init_weights=True):
    """Apply PaRaRankReduction to the model."""
    # Define configuration
    config = LoraConfig(
        r=rank,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        init_lora_weights=init_weights,
    )
    # Adding kwargs for truncated SVD
    config.truncated_svd = True
    
    # Apply PaRaRankReduction to the model
    model = make_para_rank_adapter(model, config)
    
    return model
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
def main():
    import os
    from transformers import AutoModelForCausalLM
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # Load a pre-trained model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    requires_grad(model, False)
    # Apply PaRaRankReduction
    print("Applying PaRaRankReduction...")
    # For Llama models, the attention projection modules have different names
    target_modules = None
    model = apply_para_rank_to_model(model, target_modules=target_modules, rank=16)
    
    print("PaRaRankReduction applied successfully!")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    main()