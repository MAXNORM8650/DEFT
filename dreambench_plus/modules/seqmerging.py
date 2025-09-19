import os
import torch
import copy
import torch.nn as nn
import numpy as np
from transformers.pytorch_utils import Conv1D
from peft.tuners.lora import LoraConfig
from .pepara import PaRaRankReductionLayer, make_para_rank_adapter
from .utils import load_para_rank_model

def merge_para_models_sequential(base_model, model1_path, model2_path, device="cuda"):
    # Load configurations and states for both models
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
    
    # Extract valid keys from the config for both models
    filtered_config1 = {k: v for k, v in config1.items() if k in valid_lora_config_keys}
    filtered_config2 = {k: v for k, v in config2.items() if k in valid_lora_config_keys}
    
    # Load model1 and apply it to the base model
    model1_para = make_para_rank_adapter(copy.deepcopy(base_model), LoraConfig(**filtered_config1))
    model1 = load_para_rank_model(model1_para, model1_path, device=device)
    
    # Extract Q matrices (B matrices) from model1
    q_matrices_model1 = {}
    for name, module in model1.named_modules():
        if isinstance(module, PaRaRankReductionLayer):
            q_matrices_model1[name] = module.B.detach().clone()
    
    # Load model2 and apply it to the base model
    model2_para = make_para_rank_adapter(copy.deepcopy(base_model), LoraConfig(**filtered_config2))
    model2 = load_para_rank_model(model2_para, model2_path, device=device)
    
    # Extract Q matrices from model2
    q_matrices_model2 = {}
    for name, module in model2.named_modules():
        if isinstance(module, PaRaRankReductionLayer):
            q_matrices_model2[name] = module.B.detach().clone()
    
    # Merge Q matrices: concatenate Q1 and Q2
    for name in set(q_matrices_model1.keys()) | set(q_matrices_model2.keys()):
        if name in q_matrices_model1 and name in q_matrices_model2:
            Q1 = q_matrices_model1[name]
            Q2 = q_matrices_model2[name]
            
            # Concatenate Q1 and Q2 to form Q_m
            Q_m = torch.cat([Q1, Q2], dim=1)
            
            # Perform QR decomposition to get the orthonormal matrix Q'_m
            Q_prime_m, R_prime_m = torch.linalg.qr(Q_m)
            
            # Now create a new PaRaRankReductionLayer with updated Q_prime_m
            module = model1.get_submodule(name)
            in_features = module.in_features
            out_features = module.out_features
            
            # Update the B matrix with Q'_m and the rank
            module.B.data = Q_prime_m
            module.r = Q_prime_m.shape[1]
    
    return model1  # Return the merged model