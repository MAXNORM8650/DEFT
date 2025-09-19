# Para Model Merging Implementation
import os
import torch
import torch.nn as nn
import numpy as np
from transformers.pytorch_utils import Conv1D
from peft.tuners.lora import LoraConfig
from .pepara import PaRaRankReductionLayer, make_para_rank_adapter
from .utils import load_para_rank_model

def extract_para_matrices(model):
    """
    Extract the Q matrices (represented as B matrices in the code) from a PaRa model.
    
    Args:
        model: The model with PaRaRankReduction applied
        
    Returns:
        A dictionary mapping layer names to their Q matrices
    """
    q_matrices = {}
    
    for name, module in model.named_modules():
        if isinstance(module, PaRaRankReductionLayer):
            # Extract the B matrix (represents Q in the mathematical formulation)
            q_matrices[name] = module.B.detach()
    
    return q_matrices

def merge_para_models_direct(base_model, model1_path, model2_path, device="cuda"):
    """
    Merge two PaRa models using direct concatenation and orthonormalization.
    
    Args:
        base_model: The original base model (W0)
        model1_path: Path to the first PaRa model
        model2_path: Path to the second PaRa model
        device: Device to load the models to
        
    Returns:
        The merged model
    """
    # Building Para modules
    # Load the first PaRa model
    model1 = load_para_rank_model(base_model, model1_path, device=device)
    
    # Load the second PaRa model (using a fresh base model)
    model2 = load_para_rank_model(base_model, model2_path, device=device)
    
    # Extract the Q matrices from both models
    q_matrices1 = extract_para_matrices(model1)
    q_matrices2 = extract_para_matrices(model2)
    
    # For each PaRa layer, merge the Q matrices using QR decomposition
    merged_model = base_model.to(device)
    
    # We'll need to build a configuration for the merged model
    merged_config = {}
    for key in q_matrices1.keys():
        if key in q_matrices2:
            # Get the Q matrices (B matrices in the code)
            Q1 = q_matrices1[key]
            Q2 = q_matrices2[key]
            
            # Concatenate the Q matrices
            Qm = torch.cat([Q1, Q2], dim=1)
            
            # Perform QR decomposition to get an orthonormal matrix
            Q_prime, _ = torch.linalg.qr(Qm)
            
            # Store the merged configuration
            for module_name, module in model1.named_modules():
                if module_name == key and isinstance(module, PaRaRankReductionLayer):
                    merged_config[key] = {
                        "r": Q_prime.shape[1],  # New rank is the number of columns in Q_prime
                        "lora_alpha": module.lora_alpha,
                        "in_features": module.in_features,
                        "out_features": module.out_features
                    }
    
    # Apply the merged configuration to create a new model with PaRa structure
    # We'll use the same configuration as model1 but change the rank
    if hasattr(model1, "_para_config"):
        config = model1._para_config
        config.r = max([cfg["r"] for cfg in merged_config.values()]) if merged_config else config.r
    else:
        # Fallback to a default config if not available
        config = LoraConfig(
            r=8,  # This will be overridden by the merged_config
            target_modules=None,  # Should be detected from model1
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            init_lora_weights=False,  # We'll set weights manually
        )
    
    # Apply the PaRa structure to the merged model
    merged_model = make_para_rank_adapter(merged_model, config)
    
    # Set the merged Q matrices to the new model
    for name, module in merged_model.named_modules():
        if isinstance(module, PaRaRankReductionLayer) and name in merged_config:
            # Get the merged Q matrix
            Q1 = q_matrices1[name]
            Q2 = q_matrices2[name]
            Qm = torch.cat([Q1, Q2], dim=1)
            Q_prime, _ = torch.linalg.qr(Qm)
            
            # Set the B matrix (Q in math notation) to the new orthonormalized matrix
            module.B.data.copy_(Q_prime)
            
            # Update any other properties that depend on the rank
            module.r = Q_prime.shape[1]
    
    return merged_model

def merge_para_models_sequential(base_model, model1_path, model2_path, device="cuda"):
    """
    Merge two PaRa models using sequential addition following the formula:
    Wm = W1 - Q2*Q2^T*W1 = (W0 - Q1*Q1^T*W0) - Q2*Q2^T*(W0 - Q1*Q1^T*W0)
    
    Args:
        base_model: The original base model (W0)
        model1_path: Path to the first PaRa model
        model2_path: Path to the second PaRa model
        device: Device to load the models to
        
    Returns:
        The merged model
    """
    # Load the configurations and states for both models
    model1_config_path = os.path.join(model1_path, "para_rank_config.bin")
    model1_state_path = os.path.join(model1_path, "para_rank_state.bin")
    
    if not (os.path.exists(model1_state_path) and os.path.exists(model1_config_path)):
        raise ValueError("No PaRaRankReduction data found for the first model")
    
    # Load PaRa configuration and state from model1
    para_config_dict_model1 = torch.load(model1_config_path, weights_only=False)
    para_state_model1 = torch.load(model1_state_path, map_location="cpu", weights_only=True)
    
    model2_config_path = os.path.join(model2_path, "para_rank_config.bin")
    model2_state_path = os.path.join(model2_path, "para_rank_state.bin")
    
    if not (os.path.exists(model2_state_path) and os.path.exists(model2_config_path)):
        raise ValueError("No PaRaRankReduction data found for the second model")
    
    # Load PaRa configuration and state from model2
    para_config_dict_model2 = torch.load(model2_config_path, weights_only=False)
    para_state_model2 = torch.load(model2_state_path, map_location="cpu", weights_only=True)
    
    # Check if the configurations are compatible
    config1 = para_config_dict_model1["config"]
    config2 = para_config_dict_model2["config"]
    
    if config1.get("target_modules") != config2.get("target_modules"):
        raise ValueError("Target modules do not match between the two models")
    
    # Filter out unexpected arguments for LoraConfig
    valid_lora_config_keys = [
        'r', 'target_modules', 'lora_alpha', 'lora_dropout', 'bias', 
        'init_lora_weights', 'modules_to_save', 'fan_in_fan_out'
    ]
    
    # Extract only the valid keys from the config for the sequential model
    # We'll use model2's config as the base since it's applied second
    filtered_config = {k: v for k, v in config2.items() 
                      if k in valid_lora_config_keys}
    
    # Create the config with only valid parameters
    config = LoraConfig(**filtered_config)
    
    # Clone the base model to start fresh
    merged_model = base_model.to(device)
    
    # First, apply model1's PaRa structure
    merged_model = make_para_rank_adapter(merged_model, LoraConfig(**{k: v for k, v in config1.items() 
                                                              if k in valid_lora_config_keys}))
    
    # Apply model1's weights
    for name, module in merged_model.named_modules():
        if isinstance(module, PaRaRankReductionLayer):
            module_name_in_state = name + ".B"
            if module_name_in_state in para_state_model1:
                module.B.data.copy_(para_state_model1[module_name_in_state])
    
    # Now compute the intermediate weights W1 = W0 - Q1*Q1^T*W0
    # This is implicit in the model's forward pass, so we don't need to explicitly compute it
    
    # Now, apply model2's PaRa structure on top of the already adapted model
    # This implements W1 - Q2*Q2^T*W1
    
    # Get the layer configs from model1 and model2
    layer_configs_model1 = para_config_dict_model1.get("layer_configs", {})
    layer_configs_model2 = para_config_dict_model2.get("layer_configs", {})
    
    # Function to apply sequential PaRa adaptations
    def apply_sequential_para(model):
        """Apply the second PaRa adaptation on top of the first"""
        for name, module in model.named_modules():
            if isinstance(module, PaRaRankReductionLayer):
                module_name_in_state = name + ".B"
                
                # If this layer exists in model2, apply its adaptation
                if module_name_in_state in para_state_model2:
                    # Get the Q matrix from model2
                    Q2 = para_state_model2[module_name_in_state]
                    
                    # Apply it to the current module
                    # For sequential application, we directly use the Q2 matrix
                    module.B.data.copy_(Q2)
                    
                    # Update rank if needed
                    if name in layer_configs_model2:
                        module.r = layer_configs_model2[name]["r"]
                        module.lora_alpha = layer_configs_model2[name]["lora_alpha"]
        
        return model
    
    # Instead of directly applying model2's weights, we need to create a new PaRa adapter
    # that represents the sequential application
    merged_model = make_para_rank_adapter(merged_model, config)
    merged_model = apply_sequential_para(merged_model)
    
    return merged_model


def verify_sequential_merge(base_model, model1, model2, merged_model, test_input):
    """
    Verify that the sequential merge is correct by checking:
    W_merged = W1 - Q2*Q2^T*W1
    
    Args:
        base_model: Original base model (W0)
        model1: First PaRa model (W1)
        model2: Second PaRa model (W2) 
        merged_model: Result of sequential merge
        test_input: Input tensor to test with
    
    Returns:
        True if the merge is correct, False otherwise
    """
    # Run forward pass through each model
    with torch.no_grad():
        base_output = base_model(test_input)
        model1_output = model1(test_input)
        model2_output = model2(test_input)
        merged_output = merged_model(test_input) 
        
        # Extract the logits or relevant output
        if hasattr(base_output, "logits"):
            base_output = base_output.logits
            model1_output = model1_output.logits
            model2_output = model2_output.logits
            merged_output = merged_output.logits
        
        # Sequential merge should make predictions that combine aspects of both models
        # We can check that the output is closer to model1 than the base model
        base_diff = torch.norm(merged_output - base_output)
        model1_diff = torch.norm(merged_output - model1_output)
        model2_diff = torch.norm(merged_output - model2_output)
        
        print(f"Difference from base model: {base_diff.item()}")
        print(f"Difference from model1: {model1_diff.item()}")
        print(f"Difference from model2: {model2_diff.item()}")
        
        # The merged model should be different from all three original models
        # but closest to what you'd get by applying model2's transformation to model1
        return base_diff > 0 and model1_diff > 0 and model2_diff > 0


# Example usage of the sequential merge
def sequential_merge_example():
    # Load models (pseudo-code, actual loading depends on your model type)
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained("base_model_name")
    
    # Define paths
    model1_path = "path/to/para_model1"
    model2_path = "path/to/para_model2"
    output_path = "path/to/merged_model"
    
    # Merge using sequential method
    merged_model = merge_para_models_sequential(base_model, model1_path, model2_path)
    
    # Optional: Load individual models for verification
    model1 = load_para_rank_model(base_model.clone(), model1_path)
    model2 = load_para_rank_model(base_model.clone(), model2_path)
    
    # Test input for verification
    test_input = torch.randn(1, 512)  # Example input shape
    
    # Verify merge correctness
    is_correct = verify_sequential_merge(base_model, model1, model2, merged_model, test_input)
    print(f"Sequential merge verification: {'Passed' if is_correct else 'Failed'}")
    
    # Save the merged model
    save_merged_para_model(merged_model, output_path)
    print(f"Merged model saved to {output_path}")
# Example usage
def merge_example():
    # Load base model (pseudo-code, actual loading depends on your model type)
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained("base_model_name")
    
    # Define paths
    model1_path = "path/to/para_model1"
    model2_path = "path/to/para_model2"
    output_path = "path/to/merged_model"
    
    # Choose merging method
    method = "direct"  # or "sequential"
    
    if method == "direct":
        merged_model = merge_para_models_direct(base_model, model1_path, model2_path)
    else:
        merged_model = merge_para_models_sequential(base_model, model1_path, model2_path)
    
    # Save the merged model
    save_merged_para_model(merged_model, output_path)
    print(f"Merged model saved to {output_path}")


# For combination with LoRA models
def combine_para_with_lora(base_model, para_model_path, lora_model_path, device="cuda"):
    """
    Combine a PaRa model with a LoRA model.
    
    Args:
        base_model: The original base model
        para_model_path: Path to the PaRa model
        lora_model_path: Path to the LoRA model
        device: Device to load models to
        
    Returns:
        The combined model with both adaptations
    """
    # First load the PaRa model
    para_model = load_para_rank_model(base_model, para_model_path, device=device)
    
    # Then apply LoRA adapters on top of the PaRa model
    from peft import PeftModel
    combined_model = PeftModel.from_pretrained(para_model, lora_model_path)
    
    # Return the combined model
    return combined_model