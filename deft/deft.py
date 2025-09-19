import os
import types
import torch
import json
import warnings
from enum import Enum
from tqdm import tqdm
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Type
from .injection import KnowledgeInjectionLayer, InjectionMethod
from transformers import PreTrainedModel
import os

def make_knowledge_injection_adapter(model, config):
    """
    Apply Knowledge Injection to a model by directly modifying it.
    
    Args:
        model: The model to modify
        config: The configuration containing Knowledge Injection parameters
        
    Returns:
        The modified model with Knowledge Injection applied
    """
    modified_modules = set()
    
    def replace_modules(module, parent_name=""):
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            # Skip if already modified
            if full_name in modified_modules:
                continue
            if isinstance(child, nn.Identity):
                continue  # Skip processing for Identity layers
            if config.target_modules is None:
                config.target_modules = []

            is_target = False
            if not config.target_modules:

                is_target = isinstance(child, nn.Linear)
            else:

                is_target = any(target_key in full_name for target_key in config.target_modules)

            if is_target and isinstance(child, nn.Linear):
                try:
                    # Prepare config parameters
                    config_dict = config.to_dict() if hasattr(config, "to_dict") else vars(config)
                    
                    # Create injection layer
                    injection_layer = KnowledgeInjectionLayer(base_layer=child, **config_dict)
                    setattr(module, name, injection_layer)
                    modified_modules.add(full_name)
                except Exception as e:
                    print(f"Error applying Knowledge Injection to {full_name}: {e}")
            else:
                # Recursively search for target modules
                replace_modules(child, full_name)
    
    # Apply to model's LLM if it exists, otherwise the whole model ( This is manily for unified models which has llm like omnigen)
    if hasattr(model, "llm"):
        replace_modules(model.llm, "llm")
    else:
        replace_modules(model)
    
    print(f"Applied Knowledge Injection to {len(modified_modules)} modules")
    return model

def add_knowledge_injection_methods(model):
    """
    Add save_pretrained and load_pretrained methods to enable saving/loading adapters.
    
    Args:
        model: The model to augment with save/load methods
        
    Returns:
        The augmented model
    """

    def save_pretrained(self, save_directory):
        """
        Save the model's Knowledge Injection parameters.
        
        Args:
            save_directory: Where to save the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save original model config if available
        if hasattr(self, "config"):
            self.config.save_pretrained(save_directory)
        
        # Save full model state if needed (usually not necessary for adapters)
        if hasattr(self, "_original_save_pretrained"):
            self._original_save_pretrained(save_directory)
        
        # Extract Knowledge Injection parameters
        injection_state = {}
        injection_config = {}
        
        # Collect parameters from each KnowledgeInjectionLayer
        for name, module in self.named_modules():
            if isinstance(module, KnowledgeInjectionLayer):
                # Extract lora_dropout value safely
                lora_dropout_value = 0.0
                if hasattr(module, "lora_dropout"):
                    if isinstance(module.lora_dropout, nn.Dropout):
                        lora_dropout_value = module.lora_dropout.p
                    elif isinstance(module.lora_dropout, float):
                        lora_dropout_value = module.lora_dropout
                    
                # Save layer configuration
                config_dict = {
                    "r": module.r,
                    "injection_method": module.injection_method.value,
                    "lora_alpha": module.lora_alpha,
                    "lora_dropout": lora_dropout_value,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "init_scale": module.init_scale,
                    "use_gating": module.use_gating,
                    "decomposition_method": module.decomposition_method,
                    "compute_svd_each_forward": module.compute_svd_each_forward,
                }
                
                # Save trainable parameters
                for param_name in ["P", "R_new", "lora_A", "lora_B", "gate"]:
                    if hasattr(module, param_name):
                        param = getattr(module, param_name)
                        if isinstance(param, torch.Tensor):
                            injection_state[name + "." + param_name] = param.detach().cpu()
                
                # Save cached orthogonalization if available
                if hasattr(module, "Q_P_cached") and module.Q_P_cached is not None:
                    injection_state[name + ".Q_P_cached"] = module.Q_P_cached.detach().cpu()
                
                injection_config[name] = config_dict
        
        # Save Knowledge Injection state
        if injection_state:
            torch.save(injection_state, os.path.join(save_directory, "knowledge_injection_state.bin"))
        
        # Save configuration
        if injection_config:
            config_dict = {"layer_configs": injection_config}
            if hasattr(self, "_injection_config"):
                if hasattr(self._injection_config, "__dict__"):
                    config_dict["config"] = dict(self._injection_config.__dict__)
                else:
                    config_dict["config"] = dict(self._injection_config)
            
            # Save both binary and JSON formats
            torch.save(config_dict, os.path.join(save_directory, "knowledge_injection_config.bin"))
            
            # Also save readable JSON version
            try:
                import json
                with open(os.path.join(save_directory, "knowledge_injection_config.json"), 'w') as f:
                    json_dict = {}
                    for k, v in config_dict.items():
                        if isinstance(v, dict):
                            json_dict[k] = {}
                            for sk, sv in v.items():
                                if isinstance(sv, Enum):
                                    json_dict[k][sk] = sv.value
                                else:
                                    json_dict[k][sk] = sv
                        else:
                            json_dict[k] = v
                    json.dump(json_dict, f, indent=2, default=str)
            except Exception as e:
                print(f"Warning: Could not save JSON config: {e}")
    def load_knowledge_injection_adapter(model, adapter_path):
        """
        Load a Knowledge Injection adapter into the model.
        
        Args:
            model: Model to load adapter into
            adapter_path: Path to adapter directory
            
        Returns:
            Model with adapter loaded
        """
        print(f"Loading Knowledge Injection adapter from: {adapter_path}")
        
        # Check for adapter files
        injection_config_path = os.path.join(adapter_path, "knowledge_injection_config.bin")
        injection_state_path = os.path.join(adapter_path, "knowledge_injection_state.bin")
        
        if not os.path.exists(injection_config_path) or not os.path.exists(injection_state_path):
            raise FileNotFoundError(f"Knowledge Injection files not found in {adapter_path}")
        
        # Load configuration and state
        injection_config = torch.load(injection_config_path, weights_only=False)
        injection_state = torch.load(injection_state_path, weights_only=True)
        layer_configs = injection_config.get("layer_configs", {})
        
        for name, module in list(model.named_modules()):
            if isinstance(module, KnowledgeInjectionLayer) and hasattr(module, 'base_layer'):
                try:
                    replace_module(model, name, module.base_layer)
                    print(f"Restored base layer for {name}")
                except Exception as e:
                    print(f"Error restoring base layer: {e}")
        
        modified_count = 0
        for name in layer_configs:
            try:
                # Get the target module
                try:
                    target_module = get_module_by_name(model, name)
                except Exception:
                    print(f"Module {name} not found in model")
                    continue
                
                # Skip if already a KnowledgeInjectionLayer
                if isinstance(target_module, KnowledgeInjectionLayer):
                    continue
                
                # Create injection layer
                config = layer_configs[name]
                injection_method = InjectionMethod(config["injection_method"]) \
                    if isinstance(config.get("injection_method"), str) else config.get("injection_method")
                
                # Create with safety checks
                kwargs = {
                    "base_layer": target_module,
                    "r": config["r"],
                    "injection_method": injection_method,
                    "lora_alpha": float(config.get("lora_alpha", 8.0)),
                    "init_scale": float(config.get("init_scale", 1.0)),
                    "use_gating": bool(config.get("use_gating", False)),
                    "decomposition_method": config.get("decomposition_method", None),
                    "compute_svd_each_forward": bool(config.get("compute_svd_each_forward", False)),
                }
                
                # Handle lora_dropout separately due to the Identity issue
                if "lora_dropout" in config:
                    lora_dropout = float(config["lora_dropout"])
                    if lora_dropout > 0:
                        kwargs["lora_dropout"] = lora_dropout
                
                injection_layer = KnowledgeInjectionLayer(**kwargs)
                
                # Load parameters
                for param_name in ["P", "R_new", "lora_A", "lora_B", "gate"]:
                    full_param_name = f"{name}.{param_name}"
                    if hasattr(injection_layer, param_name) and full_param_name in injection_state:
                        try:
                            getattr(injection_layer, param_name).data.copy_(injection_state[full_param_name])
                        except Exception as e:
                            print(f"Error loading parameter {param_name}: {e}")
                
                # Load cached orthogonalization
                if f"{name}.Q_P_cached" in injection_state:
                    injection_layer.Q_P_cached = injection_state[f"{name}.Q_P_cached"]
                
                # Replace module
                replace_module(model, name, injection_layer)
                modified_count += 1
                
            except Exception as e:
                print(f"Error processing layer {name}: {e}")
        
        print(f"Successfully loaded Knowledge Injection into {modified_count} layers")
        model._knowledge_injection_applied = True
        model._injection_config = injection_config.get("config", {})
        
        return model
    
    # Helper functions
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
    
    def get_module_by_name(model, name):
        """Get a module from the model by its name."""
        name_parts = name.split('.')
        if len(name_parts) == 1:
            return getattr(model, name_parts[0])
        
        module = model
        for part in name_parts:
            module = getattr(module, part)
        
        return module
    
    # Add method for getting trainable parameters
    def get_trainable_parameters(self):
        """Get parameters that require gradients for training."""
        return [p for n, p in self.named_parameters() 
                if any(x in n for x in ["P", "R_new", "lora_A", "lora_B", "gate"]) and p.requires_grad]
    
    # Attach methods to model
    model.save_pretrained = types.MethodType(save_pretrained, model)
    model.get_trainable_parameters = types.MethodType(get_trainable_parameters, model)
    model.load_knowledge_injection_adapter = staticmethod(load_knowledge_injection_adapter)
    
    # Preserve original save_pretrained if needed
    if hasattr(model, "save_pretrained") and model.save_pretrained.__func__ != save_pretrained:
        model._original_save_pretrained = model.save_pretrained
    
    return model


class KnowledgeInjectionConfig:
    def __init__(self, r=8, injection_method="project_replace", target_modules=None, 
                 lora_alpha=8.0, lora_dropout=0.0, init_scale=1.0, 
                 use_gating=False, decomposition_method="qr", 
                 compute_svd_each_forward=False):
        self.r = r
        self.injection_method = injection_method
        self.target_modules = target_modules
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.init_scale = init_scale
        self.use_gating = use_gating
        self.decomposition_method = decomposition_method
        self.compute_svd_each_forward = compute_svd_each_forward
    
    def to_dict(self):
        return vars(self)

# Example usage
if __name__ == "__main__":
    import torch.nn as nn
    import copy
    from OmniGen import OmniGenPipeline
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
    breakpoint()
    config = KnowledgeInjectionConfig(r=16, injection_method="residual_projection", target_modules=None, use_gating=True)
    adapted_model = make_knowledge_injection_adapter(pipe.model, config)
    adapted_model = add_knowledge_injection_methods(adapted_model)
    adapted_model.save_pretrained("./knowledge_injection_adapter")
    breakpoint()
    pipe2 = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
    new_model = add_knowledge_injection_methods(pipe2.model)
    new_model = new_model.load_knowledge_injection_adapter(new_model, "/nvme-data/Komal/documents/results/DREAMBOOT/inj/full/qr/checkpoints/0002000/")
    print("Adapted model saved and loaded successfully!")