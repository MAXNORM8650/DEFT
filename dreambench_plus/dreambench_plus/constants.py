class IdentityDict(dict):

    def __missing__(self, key):
        if key is None:
            return None
        return key


MODEL_ZOOS = IdentityDict(
    {
        "huggingface/model_name_or_path": "path/to/snapshots",
        # ...
    }
)
PATH_TO_MODEL_NAME = IdentityDict({v: k for k, v in MODEL_ZOOS.items()})

LOCAL_FILES_ONLY = False

OPENAI_BASE_URL = None
OPENAI_API_KEYS = [
    "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",
    # "sk-xxx2",
]
OPENAI_MODEL = "gpt-4o-2024-05-13"

# fmt: off
METHODS = {
    # "Textual Inversion SD"      : "samples/textual_inversion_sd_gs7_5_step100_seed42_torch_float16",
    # "DreamBooth SD"             : "samples/dreambooth_sd_gs7_5_step100_seed42_torch_float16",
    "DreamBooth LoRA SDXL"      : "samples/dreambooth_lora_sdxl_gs7_5_step100_seed42_torch_float16",
    "DreamBooth deft SDXL"      : "samples/dreambooth_deft_GFR8_sdxl_gs7_5_step100_seed42_torch_float16",
    # "DreamBooth deftp SDXL"      : "samples/dreambooth_deftp_sdxl_gs7_5_step100_seed42_torch_float16",
    "DreamBooth para SDXL"      : "samples/dreambooth_para_sdxl_gs7_5_step100_seed42_torch_float16",
    "BLIP-Diffusion"            : "samples/blip_diffusion_gs7_5_step100_seed42_torch_float16",
    # "Emu2"                      : "samples/emu2_gs3_step50_seed42_torch_bfloat16",
    # "IP-Adapter-Plus ViT-H SDXL": "samples/ip_adapter_plus_vit_h_sdxl_scale0_6_gs7_5_step100_seed42_torch_float16",
    # "IP-Adapter ViT-G SDXL"     : "samples/ip_adapter_vit_g_sdxl_scale0_6_gs7_5_step100_seed42_torch_float16",
}
# fmt: on

DREAMBENCH_DIR = None
DREAMBENCH_PLUS_DIR = "/nvme-data/Komal/documents/omni_datasets/dreambench_plus_data/data"
HUMAN_RATING_RAW_DATA = "/nvme-data/Komal/documents/omni_datasets/dreambench_plus_data/data_human_rating/raw_data"
HUMAN_RATING_MERGED_DATA = "/nvme-data/Komal/documents/omni_datasets/dreambench_plus_data/data_human_rating/merged_data"

GPT_RATING_DATA = "/nvme-data/Komal/documents/omni_datasets/dreambench_plus_data/data_gpt_rating"


# python --method "DreamBooth deftp SDXLL" \
# --out_dir data_gpt_rating/concept_preservation_full/dreambooth_deftp_sdxl \
# --category subject \
# --ablation_settings full