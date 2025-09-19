#!/bin/bash

METHODS=("para" "lora" "deft_GFR4")
STEPS=(200 300 400)

for method in "${METHODS[@]}"; do
    for number in "${STEPS[@]}"; do
        echo "Running with method: $method and steps: $number"
        CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
        --nproc_per_node=1 \
        --nnodes=1 \
        --master_port=29507 \
        generate_images.py \
        --method dreambooth_${method}_sdxl \
        --use_default_params True \
        --db_or_ti_output_dir /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/submodules/dreambench_plus/work_dirs/dreambench_plus/dreambooth_${method}_sdxl_progress \
        --steps $number
    done
done