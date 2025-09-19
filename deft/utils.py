# #import all necessary libraries
import os
import torch    
import torch.nn as nn
import types
from transformers.pytorch_utils import Conv1D
from peft.tuners.lora import LoraConfig
from .pepara import PaRaRankReductionLayer


# def save_para_rank_model(model, save_directory):
#     """
#     Save a model with PaRaRankReduction layers to disk.
    
#     Args:
#         model: The model with PaRaRankReduction applied
#         save_directory: Directory to save the model
#     """
#     os.makedirs(save_directory, exist_ok=True)
    
#     # Save the original model weights
#     model.save_pretrained(save_directory)
    
#     # Save PaRa-specific weights separately
#     para_state = {}
#     para_config = {}
    
#     # Extract PaRa B matrices and config
#     for name, module in model.named_modules():
#         if isinstance(module, PaRaRankReductionLayer):
#             # Save B matrix for this layer
#             para_state[name + ".B"] = module.B.detach().cpu()
            
#             # Save config for this layer
#             para_config[name] = {
#                 "r": module.r,
#                 "lora_alpha": module.lora_alpha,
#                 "in_features": module.in_features,
#                 "out_features": module.out_features
#             }
    
#     # Save the PaRa-specific state
#     torch.save(para_state, os.path.join(save_directory, "para_rank_state.bin"))
    
#     # Save the configuration
#     if hasattr(model, "_para_config"):
#         config_dict = {
#             "config": model._para_config.__dict__,
#             "layer_configs": para_config
#         }
#         torch.save(config_dict, os.path.join(save_directory, "para_rank_config.bin"))

# def load_para_rank_model(model, model_path, device=None):
#     """
#     Load a model with PaRaRankReduction from disk.
    
#     Args:
#         model: The base model to apply PaRa to (already loaded)
#         model_path: Path to the saved PaRa configuration
#         device: Device to load the model to
    
#     Returns:
#         The loaded model with PaRaRankReduction applied
#     """
#     # Check if this is a PaRa model
#     para_state_path = os.path.join(model_path, "para_rank_state.bin")
#     para_config_path = os.path.join(model_path, "para_rank_config.bin")
    
#     if not (os.path.exists(para_state_path) and os.path.exists(para_config_path)):
#         print("No PaRaRankReduction data found, returning regular model")
#         return model

    
#     # Load PaRa configuration
#     para_config_dict = torch.load(para_config_path, weights_only=False)
    
#     # Filter out unexpected arguments for LoraConfig
#     valid_lora_config_keys = [
#         'r', 'target_modules', 'lora_alpha', 'lora_dropout', 'bias', 
#         'init_lora_weights', 'modules_to_save', 'fan_in_fan_out'
#     ]
    
#     # Extract only the valid keys from the config
#     filtered_config = {k: v for k, v in para_config_dict["config"].items() 
#                       if k in valid_lora_config_keys}
    
#     # Create the config with only valid parameters
#     config = LoraConfig(**filtered_config)
#     # Apply PaRaRankReduction structure to the model
#     # model = make_para_rank_adapter(model, config)
#     # Load PaRa-specific weights
#     para_state = torch.load(para_state_path, map_location="cpu", weights_only=True)
    
#     # Apply the loaded weights
#     for name, module in model.named_modules():
#         if isinstance(module, PaRaRankReductionLayer):
#             if name + ".B" in para_state:
#                 module.B.data.copy_(para_state[name + ".B"])
    
#     if device is not None:
#         model = model.to(device)
    
#     return model

# def add_para_rank_methods(model):
#     """Add save_pretrained method to the model."""
    
#     def save_pretrained(self, save_directory):
#         os.makedirs(save_directory, exist_ok=True)
        
#         # Save the original model using its save method if available
#         if hasattr(self, "config"):
#             self.config.save_pretrained(save_directory)
        
#         # Save model weights
#         if hasattr(self, "save_pretrained") and self.save_pretrained.__func__ != save_pretrained:
#             # Call parent implementation first
#             original_save = self.save_pretrained
#             original_save(save_directory)
#         else:
#             # Fallback to state dict saving
#             torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
#         # Save PaRa-specific weights separately
#         para_state = {}
#         para_config = {}
        
#         # Extract PaRa B matrices and config
#         for name, module in self.named_modules():
#             if isinstance(module, PaRaRankReductionLayer):
#                 # Save B matrix for this layer
#                 para_state[name + ".B"] = module.B.detach().cpu()
                
#                 # Save config for this layer
#                 para_config[name] = {
#                     "r": module.r,
#                     "lora_alpha": module.lora_alpha,
#                     "in_features": module.in_features,
#                     "out_features": module.out_features
#                 }
        
#         # Save the PaRa-specific state
#         torch.save(para_state, os.path.join(save_directory, "para_rank_state.bin"))
        
#         # Save the configuration
#         if hasattr(self, "_para_config"):
#             config_dict = {
#                 "config": self._para_config.__dict__,
#                 "layer_configs": para_config
#             }
#             torch.save(config_dict, os.path.join(save_directory, "para_rank_config.bin"))
    
#     # Add the methods to the model
#     model.save_pretrained = types.MethodType(save_pretrained, model)
    
#     return model
# def apply_para_rank_to_model(model, target_modules=None, rank=8, lora_alpha=8, lora_dropout=0.1, init_weights=True):
#     """Apply PaRaRankReduction to the model."""
#     # Define configuration
#     config = LoraConfig(
#         r=rank,
#         target_modules=target_modules,
#         lora_alpha=lora_alpha,
#         lora_dropout=lora_dropout,
#         bias="none",
#         init_lora_weights=init_weights,
#     )
    
#     # Apply PaRaRankReduction to the model
#     model = make_para_rank_adapter(model, config)
    
#     # Add save/load methods
#     model = add_para_rank_methods(model)
    
#     return model



#### New function for utility

import os
import types
import torch
import json
import warnings
from enum import Enum
from typing import Dict, Any, Optional, Union, Type
from .genpara import GeneralizedPaRaRankReductionLayer
from .para import PaRaRankReductionLayer
from transformers import PreTrainedModel

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

def load_para_rank_adapter(base_model, adapter_path, lora_config=None):
    """
    Load a PaRa-Rank adapter into a model.
    
    Args:
        base_model: The base model to load the adapter into
        adapter_path: Path to the adapter weights
        lora_config: Optional LoRA configuration for new models
        
    Returns:
        The model with the adapter loaded
    """
    print(f"Loading PaRa-Rank adapter from: {adapter_path}")
    
    # First prepare the model with PaRa-Rank adapter structure if lora_config is provided
    if lora_config is not None:
        model = make_para_rank_adapter(base_model, lora_config)
        model = add_para_rank_methods(model)
    else:
        # If no config provided, just add the methods to the existing model
        model = add_para_rank_methods(base_model)
    
    # Check if we have a PaRa-Rank adapter or a regular adapter
    para_config_path = os.path.join(adapter_path, "para_rank_config.bin")
    para_state_path = os.path.join(adapter_path, "para_rank_state.bin")
    
    if os.path.exists(para_config_path) and os.path.exists(para_state_path):
        # We have a PaRa-Rank adapter, load it using our custom method
        print("Found PaRa-Rank adapter, loading with specialized loader...")
        
        # If model is a class that needs instantiation, handle that case
        if isinstance(model, type):
            # This is a class, not an instance
            return model.load_pretrained(adapter_path)
        else:
            # This is already an instance, use a different approach
            # Load PaRa-Rank configuration
            para_config = torch.load(para_config_path, weights_only=False)
            layer_configs = para_config.get("layer_configs", {})
            
            # Load PaRa-Rank state
            para_state = torch.load(para_state_path, weights_only=True)
            
            # Apply PaRa-Rank parameters to matching layers
            for name, module in model.named_modules():
                if name in layer_configs:
                    config = layer_configs[name]
                    
                    # Load parameters based on layer type
                    if isinstance(module, (PaRaRankReductionLayer, GeneralizedPaRaRankReductionLayer)):
                        # Layer already has the right type, just load parameters
                        if hasattr(module, "sparse_parameterization") and module.sparse_parameterization:
                            if name + ".sparse_mask" in para_state:
                                module.register_buffer("sparse_mask", para_state[name + ".sparse_mask"])
                            if name + ".B_dense" in para_state:
                                module.B_dense.data.copy_(para_state[name + ".B_dense"])
                        elif name + ".B" in para_state:
                            module.B.data.copy_(para_state[name + ".B"])
                        
                        if hasattr(module, "W") and name + ".W" in para_state:
                            module.W.data.copy_(para_state[name + ".W"])
                        
                        if hasattr(module, "H") and name + ".H" in para_state:
                            module.H.data.copy_(para_state[name + ".H"])
                    else:
                        # Need to create the right layer type and replace it
                        print(f"Creating PaRa-Rank layer for {name}")
                        
                        # Create appropriate layer based on config
                        if "decomposition_method" in config:
                            # Generalized layer
                            para_layer = GeneralizedPaRaRankReductionLayer(
                                base_layer=module,
                                r=config["r"],
                                lora_alpha=config["lora_alpha"],
                                decomposition_method=config.get("decomposition_method", "qr"),
                                adaptive_rank=config.get("adaptive_rank", False),
                                variance_threshold=config.get("variance_threshold", 0.95),
                                sparse_parameterization=config.get("sparse_parameterization", False),
                                sparsity_factor=config.get("sparsity_factor", 0.1)
                            )
                        else:
                            # Original layer
                            para_layer = PaRaRankReductionLayer(
                                base_layer=module,
                                r=config["r"],
                                lora_alpha=config["lora_alpha"],
                                truncated_svd=config.get("truncated_svd", False)
                            )
                        
                        # Load parameters
                        if hasattr(para_layer, "sparse_parameterization") and para_layer.sparse_parameterization:
                            if name + ".sparse_mask" in para_state:
                                para_layer.register_buffer("sparse_mask", para_state[name + ".sparse_mask"])
                            if name + ".B_dense" in para_state:
                                para_layer.B_dense.data.copy_(para_state[name + ".B_dense"])
                        elif name + ".B" in para_state:
                            para_layer.B.data.copy_(para_state[name + ".B"])
                        
                        if hasattr(para_layer, "W") and name + ".W" in para_state:
                            para_layer.W.data.copy_(para_state[name + ".W"])
                        
                        if hasattr(para_layer, "H") and name + ".H" in para_state:
                            para_layer.H.data.copy_(para_state[name + ".H"])
                        
                        # Replace module in model
                        replace_module(model, name, para_layer)
            
            return model
    else:
        # Regular adapter (not PaRa-Rank), load using standard methods
        print("No PaRa-Rank adapter found, loading weights as standard adapter...")
        
        # Load the adapter using the model's own methods or state dict loading
        adapter_path_weights = os.path.join(adapter_path, "pytorch_model.bin") 
        if os.path.exists(adapter_path_weights):
            state_dict = torch.load(adapter_path_weights, weights_only=True)
            # Load state dict with standard PyTorch mechanisms
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded adapter with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
        
        return model


def replace_module(model, name, new_module):
    """Replace a module in the model with a new module."""
    name_parts = name.split('.')
    if len(name_parts) == 1:
        setattr(model, name_parts[0], new_module)
        return
    
    parent = model
    for part in name_parts[:-1]:
        parent = getattr(parent, part)
    
    setattr(parent, name_parts[-1], new_module)
def add_para_rank_methods(model):
    """
    Add save_pretrained and load_pretrained methods to the model.
    
    Args:
        model: The model to augment with save/load methods
        
    Returns:
        The augmented model
    """
    def save_pretrained(self, save_directory):
        """
        Save the model with its PaRa-Rank specific parameters.
        
        Args:
            save_directory: Directory where to save the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the original model using its save method if available
        if hasattr(self, "config"):
            self.config.save_pretrained(save_directory)
        
        # Save model weights
        if hasattr(self, "_original_save_pretrained"):
            # Call original implementation
            self._original_save_pretrained(save_directory)
        else:
            # Fallback to state dict saving
            torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save PaRa-specific weights separately
        para_state = {}
        para_config = {}
        
        # Extract PaRa parameters and config for each layer
        for name, module in self.named_modules():
            if isinstance(module, (PaRaRankReductionLayer, GeneralizedPaRaRankReductionLayer)):
                # Basic parameters common to all variants
                config_dict = {
                    "r": module.r,
                    "lora_alpha": module.lora_alpha,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                }
                
                # Add method-specific parameters
                if hasattr(module, "decomposition_method"):
                    config_dict["decomposition_method"] = module.decomposition_method.value
                
                if hasattr(module, "adaptive_rank"):
                    config_dict["adaptive_rank"] = module.adaptive_rank
                    if module.adaptive_rank:
                        config_dict["variance_threshold"] = module.variance_threshold
                
                if hasattr(module, "sparse_parameterization"):
                    config_dict["sparse_parameterization"] = module.sparse_parameterization
                    if module.sparse_parameterization:
                        config_dict["sparsity_factor"] = module.sparsity_factor
                        # Save sparse mask
                        para_state[name + ".sparse_mask"] = module.sparse_mask.detach().cpu()
                
                if hasattr(module, "truncated_svd"):
                    config_dict["truncated_svd"] = module.truncated_svd
                
                # Save the appropriate parameters based on the module type
                if hasattr(module, "B"):
                    para_state[name + ".B"] = module.B.detach().cpu()
                
                if hasattr(module, "B_dense"):
                    para_state[name + ".B_dense"] = module.B_dense.detach().cpu()
                
                # Save NMF-specific parameters if they exist
                if hasattr(module, "W") and hasattr(module, "H"):
                    para_state[name + ".W"] = module.W.detach().cpu()
                    para_state[name + ".H"] = module.H.detach().cpu()
                
                # Store the config for this layer
                para_config[name] = config_dict
        
        # Save the PaRa-specific state
        if para_state:
            torch.save(para_state, os.path.join(save_directory, "para_rank_state.bin"))
        
        # Save the configuration
        if para_config:
            # Include global configuration if available
            config_dict = {"layer_configs": para_config}
            if hasattr(self, "_para_config"):
                config_dict["config"] = self._para_config.__dict__
            
            # Save as both binary for backward compatibility and JSON for readability
            torch.save(config_dict, os.path.join(save_directory, "para_rank_config.bin"))
            
            # Also save as JSON for easy inspection
            with open(os.path.join(save_directory, "para_rank_config.json"), 'w') as f:
                json.dump(config_dict, f, indent=2, default=lambda x: x.value if isinstance(x, Enum) else str(x))
    
    def load_pretrained(cls, pretrained_model_path, *args, **kwargs):
        """
        Load a model with PaRa-Rank parameters from a directory.
        
        Args:
            pretrained_model_path: Path to the directory containing model files
            *args, **kwargs: Additional arguments passed to the model constructor
            
        Returns:
            The loaded model with PaRa-Rank parameters
        """
        # First check if this is a PaRa-Rank model
        para_config_path = os.path.join(pretrained_model_path, "para_rank_config.bin")
        para_state_path = os.path.join(pretrained_model_path, "para_rank_state.bin")
        
        if not os.path.exists(para_config_path) or not os.path.exists(para_state_path):
            warnings.warn(f"No PaRa-Rank configuration found in {pretrained_model_path}. "
                         "Loading as a standard model.")
            
            # Use the model's own from_pretrained if available
            if hasattr(cls, "from_pretrained"):
                return cls.from_pretrained(pretrained_model_path, *args, **kwargs)
            else:
                # Create a new model instance
                model = cls(*args, **kwargs)
                model.load_state_dict(torch.load(os.path.join(pretrained_model_path, "pytorch_model.bin")), weights_only=True)
                return model
        
        # Load PaRa-Rank configuration
        para_config = torch.load(para_config_path, weights_only=False)
        layer_configs = para_config.get("layer_configs", {})
        
        # Load PaRa-Rank state
        para_state = torch.load(para_state_path, weights_only=True)
        
        # Create model (either from scratch or using from_pretrained)
        if hasattr(cls, "from_pretrained"):
            model = cls.from_pretrained(pretrained_model_path, *args, **kwargs)
        else:
            model = cls(*args, **kwargs)
            # Load base model weights
            model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
        
        # Apply PaRa-Rank parameters to matching layers
        for name, module in model.named_modules():
            if name in layer_configs:
                config = layer_configs[name]
                
                # Determine which PaRa-Rank class to use based on config
                if "decomposition_method" in config:
                    # Create a generalized layer
                    decomposition_method = config.get("decomposition_method", "qr")
                    adaptive_rank = config.get("adaptive_rank", False)
                    variance_threshold = config.get("variance_threshold", 0.95)
                    sparse_parameterization = config.get("sparse_parameterization", False)
                    sparsity_factor = config.get("sparsity_factor", 0.1)
                    
                    # Replace with generalized layer
                    para_layer = GeneralizedPaRaRankReductionLayer(
                        base_layer=module,
                        r=config["r"],
                        lora_alpha=config["lora_alpha"],
                        decomposition_method=decomposition_method,
                        adaptive_rank=adaptive_rank,
                        variance_threshold=variance_threshold,
                        sparse_parameterization=sparse_parameterization,
                        sparsity_factor=sparsity_factor
                    )
                    
                    # Load specific parameters
                    if sparse_parameterization and name + ".sparse_mask" in para_state:
                        para_layer.register_buffer("sparse_mask", para_state[name + ".sparse_mask"])
                    
                    if name + ".B_dense" in para_state:
                        para_layer.B_dense.data.copy_(para_state[name + ".B_dense"])
                    elif name + ".B" in para_state:
                        para_layer.B.data.copy_(para_state[name + ".B"])
                    
                    if hasattr(para_layer, "W") and name + ".W" in para_state:
                        para_layer.W.data.copy_(para_state[name + ".W"])
                    
                    if hasattr(para_layer, "H") and name + ".H" in para_state:
                        para_layer.H.data.copy_(para_state[name + ".H"])
                    
                else:
                    # Create original PaRaRankReductionLayer
                    truncated_svd = config.get("truncated_svd", False)
                    
                    para_layer = PaRaRankReductionLayer(
                        base_layer=module,
                        r=config["r"],
                        lora_alpha=config["lora_alpha"],
                        truncated_svd=truncated_svd
                    )
                    
                    # Load B matrix
                    if name + ".B" in para_state:
                        para_layer.B.data.copy_(para_state[name + ".B"])
                
                # Replace the original module with PaRa layer
                replace_module(model, name, para_layer)
        
        # Store any global PaRa config if available
        if "config" in para_config:
            model._para_config = types.SimpleNamespace(**para_config["config"])
        
        # Add the methods to the model
        model.save_pretrained = types.MethodType(save_pretrained, model)
        
        # If the model already has save_pretrained, preserve the original
        if hasattr(model, "save_pretrained") and model.save_pretrained.__func__ != save_pretrained:
            model._original_save_pretrained = model.save_pretrained
        
        return model
    
    def replace_module(model, name, new_module):
        """Replace a module in the model with a new module."""
        name_parts = name.split('.')
        if len(name_parts) == 1:
            setattr(model, name_parts[0], new_module)
            return
        
        parent_name = '.'.join(name_parts[:-1])
        child_name = name_parts[-1]
        
        parent = model
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, child_name, new_module)
    
    # Add the methods to the model
    model.save_pretrained = types.MethodType(save_pretrained, model)
    
    # Add the class method to the model class
    model.__class__.load_pretrained = classmethod(load_pretrained)
    
    # If the model already has save_pretrained, preserve the original
    if hasattr(model, "save_pretrained") and model.save_pretrained.__func__ != save_pretrained:
        model._original_save_pretrained = model.save_pretrained
    
    return model

class LoraConfigExtended(LoraConfig):
    def __init__(self, decomposition_method=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decomposition_method = decomposition_method


# Example usage
if __name__ == "__main__":
    import torch.nn as nn

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 64)
        
        def forward(self, x):
            return self.linear(x)
    
    # Create model and add PaRa layer
    model = SimpleModel()
    model.linear = GeneralizedPaRaRankReductionLayer(
        base_layer=model.linear,
        r=8,
        decomposition_method="eigen",
        adaptive_rank=True
    )
    
    # Add save/load methods
    model = add_para_rank_methods(model)
    
    # Save the model
    model.save_pretrained("./para_rank_model")
    
    # Load the model
    loaded_model = SimpleModel.load_pretrained("./para_rank_model")
    
    print("Model saved and loaded successfully!")