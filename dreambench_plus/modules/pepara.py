import math
import types
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.lora import LoraConfig
from .genpara import GeneralizedPaRaRankReductionLayer
from .para import PaRaRankReductionLayer
from .injection import KnowledgeInjectionAdapter, KnowledgeInjectionLayer

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