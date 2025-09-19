#!/bin/bash

# Example script to run the multi-GPU OmniGen code

# Path to adapter weights
ADAPTER_PATH="/nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/"

# Output directory
OUTPUT_DIR="/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/eval/visualcloze_canny_depth"

# Experiments file
EXPERIMENTS_FILE="/media/mbzuaiser/SSD1/Komal/Graph200K/VisualCloze/testomnijson.jsonl"

# Run with multiple GPUs
CUDA_VISIBLE_DEVICES=1,2,3 python evaluations/multi_gpu_omnigen.py \
    --processes_per_gpu 3 \
    --output_dir "${OUTPUT_DIR}" \
    --adapter_path "${ADAPTER_PATH}" \
    --is_parainj \
    --lora_rank 16 \
    --lora_alpha 16 \
    --decomposition_method qr \
    --max_input_image_size 1024 \
    --experiments_file "${EXPERIMENTS_FILE}"

# Note: Use --is_para for PaRa adapter or --is_parainj for PaRa injection
# CUDA_VISIBLE_DEVICES=1,2,3 python evaluations/multi_gpu_omnigen.py \
#     --processes_per_gpu 3 \
#     --output_dir "${OUTPUT_DIR}" \
#     --adapter_path "${ADAPTER_PATH}" \
#     --lora_rank 16 \
#     --lora_alpha 16 \
#     --decomposition_method qr \
#     --max_input_image_size 1024 \
#     --experiments_file "${EXPERIMENTS_FILE}"