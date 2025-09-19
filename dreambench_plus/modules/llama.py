import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import _get_submodules
from peft.tuners.lora import LoraConfig, LoraModel
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

class PaRaRankReductionConfig(LoraConfig):
    """Configuration for PaRaRankReduction method."""
    def __init__(
        self,
        r: int = 8,
        target_modules: Optional[Union[List[str], str]] = None,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        bias: str = "none",
        modules_to_save: Optional[List[str]] = None,
        init_lora_weights: bool = True,
        **kwargs,
    ):
        super().__init__(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            modules_to_save=modules_to_save,
            init_lora_weights=init_lora_weights,
            **kwargs,
        )
        self.peft_type = "PARA_RANK"


class PaRaRankReductionLayer(nn.Module):
    """Implementation of PaRaRankReduction technique."""
    
    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
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
        
        # Adapt scaling based on the dimensions and rank
        self.scaling = lora_alpha / r
        
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        
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
        
        # Apply QR decomposition to get orthogonal matrix Q
        Q, _ = torch.linalg.qr(self.B)  # Q ∈ R^(out_features × r)
        
        # Compute Q * Q^T
        QQT = torch.matmul(Q, Q.transpose(0, 1))  # Q * Q^T ∈ R^(out_features × out_features)
        
        # Apply the PaRa adjustment: W - Q*Q^T*W
        # For efficiency, we'll compute directly with the input: x @ (W - Q*Q^T*W)^T
        # Which is: x @ W^T - x @ W^T @ Q @ Q^T
        
        # Get the adjustment part: (Q*Q^T*W) @ x
        adjustment = torch.matmul(x_dropped, original_weights.transpose(0, 1))
        adjustment = torch.matmul(adjustment, QQT)
        
        # Compute the final output: original - adjustment
        return original_output - adjustment


class PaRaRankAdapter(nn.Module):
    """A proper wrapper for the PaRaRankReduction adapter."""
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self._find_and_replace()
        
    def _find_and_replace(self):
        """Find the target modules and replace them with PaRaRankReduction layers."""
        for name, module in self.model.named_modules():
            if any(target_key in name for target_key in self.config.target_modules):
                # Check if this is a leaf module (not containing other modules)
                if len(list(module.children())) == 0 and isinstance(module, nn.Linear):
                    # Get parent module
                    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                    module_name = name.rsplit(".", 1)[1] if "." in name else name
                    
                    if parent_name:
                        parent = self.model.get_submodule(parent_name)
                    else:
                        parent = self.model
                    
                    # Create PaRaRankReduction layer
                    para_layer = PaRaRankReductionLayer(
                        base_layer=module,
                        r=self.config.r,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                        init_lora_weights=self.config.init_lora_weights,
                    )
                    
                    # Replace the module
                    setattr(parent, module_name, para_layer)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def apply_para_rank_to_model(model, target_modules=None, rank=8):
    """Apply PaRaRankReduction to the model."""
    # Define configuration
    config = PaRaRankReductionConfig(
        r=rank,
        target_modules=target_modules,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        init_lora_weights=True,
    )
    
    # Apply PaRaRankReduction to the model
    model = PaRaRankAdapter(model, config)
    
    return model


def main():
    from transformers import AutoModelForCausalLM
    
    # Load a pre-trained model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    requires_grad(model, False)
    # Apply PaRaRankReduction
    print("Applying PaRaRankReduction...")
    # For Llama models, the attention projection modules have different names
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = apply_para_rank_to_model(model, target_modules=target_modules, rank=16)
    
    print("PaRaRankReduction applied successfully!")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    main()