def load_ki_into_unet(state_dict, unet):
    for name, module in unet.named_modules():
        if getattr(module, "lora_layer", None) is None:   # <- safe test
            continue

        inj = module.lora_layer
        prefix = f"{name}.lora_layer"

        inj.P.data.copy_(state_dict[f"{prefix}.P"])
        if inj.use_gating!="0":
            inj.R_new.data.copy_(state_dict[f"{prefix}.R_new"])
        if inj.use_gating:
            inj.gate.data.copy_(state_dict[f"{prefix}.gate"])
    return unet