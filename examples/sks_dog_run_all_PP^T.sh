# #!/bin/bash

# # First command with lora_rank 16
# echo "Running with lora_rank 4..."
# accelerate launch --num_processes=4 --main_process_port=29502 train_iter.py \
#     --model_name_or_path Shitao/OmniGen-v1 \
#     --batch_size_per_device 1 \
#     --condition_dropout_prob 0.01 \
#     --lr 1e-3 \
#     --use_injection \
#     --lora_rank 4 \
#     --json_file ./toy_data/toy_subject_data.jsonl \
#     --image_path ./toy_data/images \
#     --max_input_length_limit 18000 \
#     --keep_raw_resolution \
#     --max_image_size 1024 \
#     --gradient_accumulation_steps 1 \
#     --ckpt_every 100 \
#     --epochs 200 \
#     --log_every 1 \
#     --results_dir /nvme-data/Komal/documents/results/INJpara/SKSdog/QLr_R4_PP^T

# echo "Running with lora_rank 16..."
# accelerate launch --num_processes=3 --main_process_port=29502 train_iter.py \
#     --model_name_or_path Shitao/OmniGen-v1 \
#     --batch_size_per_device 1 \
#     --condition_dropout_prob 0.01 \
#     --lr 1e-3 \
#     --use_injection \
#     --lora_rank 16 \
#     --json_file ./toy_data/toy_subject_data.jsonl \
#     --image_path ./toy_data/images \
#     --max_input_length_limit 18000 \
#     --keep_raw_resolution \
#     --max_image_size 1024 \
#     --gradient_accumulation_steps 1 \
#     --ckpt_every 100 \
#     --epochs 200 \
#     --log_every 1 \
#     --results_dir /nvme-data/Komal/documents/results/INJpara/SKSdog/QLr_R16_PP^T

# # Second command with lora_rank 8
# echo "Running with lora_rank 8..."
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 --main_process_port=29502 train_iter.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --batch_size_per_device 1 \
    --condition_dropout_prob 0.01 \
    --lr 1e-3 \
    --use_injection \
    --lora_rank 8 \
    --json_file ./toy_data/toy_subject_data.jsonl \
    --image_path ./toy_data/images \
    --max_input_length_limit 18000 \
    --keep_raw_resolution \
    --max_image_size 1024 \
    --gradient_accumulation_steps 1 \
    --ckpt_every 100 \
    --epochs 200 \
    --log_every 1 \
    --results_dir /nvme-data/Komal/documents/results/INJpara/SKSdog/QLr_R8_PP^T

# # Third command with lora_rank 32
# # echo "Running with lora_rank 32..."
# # accelerate launch --num_processes=4 --main_process_port=29502 train_iter.py \
# #     --model_name_or_path Shitao/OmniGen-v1 \
# #     --batch_size_per_device 1 \
# #     --condition_dropout_prob 0.01 \
# #     --lr 1e-3 \
# #     --use_injection \
# #     --lora_rank 32 \
# #     --json_file ./toy_data/toy_subject_data.jsonl \
# #     --image_path ./toy_data/images \
# #     --max_input_length_limit 18000 \
# #     --keep_raw_resolution \
# #     --max_image_size 1024 \
# #     --gradient_accumulation_steps 1 \
# #     --ckpt_every 100 \
# #     --epochs 200 \
# #     --log_every 1 \
# #     --results_dir /nvme-data/Komal/documents/results/INJpara/SKSdog/QLr_R32
# echo "Running with lora_rank 64..."
# # accelerate launch --num_processes=4 --main_process_port=29502 train_iter.py \
# #     --model_name_or_path Shitao/OmniGen-v1 \
# #     --batch_size_per_device 1 \
# #     --condition_dropout_prob 0.01 \
# #     --lr 1e-3 \
# #     --use_injection \
# #     --lora_rank 64 \
# #     --json_file ./toy_data/toy_subject_data.jsonl \
# #     --image_path ./toy_data/images \
# #     --max_input_length_limit 18000 \
# #     --keep_raw_resolution \
# #     --max_image_size 1024 \
# #     --gradient_accumulation_steps 1 \
# #     --ckpt_every 100 \
# #     --epochs 200 \
# #     --log_every 1 \
# #     --results_dir /nvme-data/Komal/documents/results/INJpara/SKSdog/QLr_R64

# echo "Running with lora_rank 8 with all modules..."
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 --main_process_port=29502 train_iter1.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --batch_size_per_device 1 \
    --condition_dropout_prob 0.01 \
    --lr 1e-3 \
    --use_injection \
    --lora_rank 8 \
    --json_file ./toy_data/toy_subject_data.jsonl \
    --image_path ./toy_data/images \
    --max_input_length_limit 18000 \
    --keep_raw_resolution \
    --max_image_size 1024 \
    --gradient_accumulation_steps 1 \
    --ckpt_every 20 \
    --epochs 200 \
    --log_every 1 \
    --results_dir /nvme-data/Komal/documents/results/INJpara/SKSdog/QLr_R8_all_modules_PP^T


echo "All commands have been executed successfully."
# Now running evaluation command
# echo "Running evaluation..."
# python evaluations/eval_personalization.py \
#     --adapter_path /nvme-data/Komal/documents/results/INJpara/SKSdog/QLr_R16_PP^T/checkpoints/0000200/  \
#     --output_dir './results/sks/dog_abalation/QLr_R16_PP^T/'

python evaluations/eval_personalization.py \
    --adapter_path /nvme-data/Komal/documents/results/INJpara/SKSdog/QLr_R8_PP^T/checkpoints/0000200/  \
    --output_dir './results/sks/dog_abalation/QLr_R8_PP^T/'
# python evaluations/eval_personalization.py \
#     --adapter_path /nvme-data/Komal/documents/results/INJpara/SKSdog/QLr_R4_PP^T/checkpoints/0000200/  \
#     --output_dir './results/sks/dog_abalation/QLr_R4_PP^T'
# python evaluations/eval_personalization.py \
#     --adapter_path /nvme-data/Komal/documents/results/INJpara/SKSdog/QLr_R64/checkpoints/0000200/  \
#     --output_dir './results/sks/dog_abalation/QLr_R64/'
python evaluations/eval_personalization.py \
    --adapter_path /nvme-data/Komal/documents/results/INJpara/SKSdog/QLr_R8_all_modules_PP^T/checkpoints/0000200/  \
    --output_dir './results/sks/dog_abalation/QLr_R8_all_modules_PP^T/'
echo "Evaluation completed."
echo "All tasks have been completed successfully."
