# Fine-tuning OmniGen

Fine-tuning Omnigen can better help you handle specific image generation tasks. For example, by fine-tuning on a person's images, you can generate multiple pictures of that person while maintaining task consistency.

A lot of previous work focused on designing new networks to facilitate specific tasks. For instance, ControlNet was proposed to handle image conditions, and IP-Adapter was constructed to maintain ID features. If you want to perform new tasks, you need to build new architectures and repeatedly debug them. Adding and adjusting extra network parameters is usually time-consuming and labor-intensive, which is not user-friendly and cost-efficient enough. However, with Omnigen, all of this becomes very simple.

By comparison, Omnigen can accept multi-modal conditional inputs and has been pre-trained on various tasks. You can fine-tune it on any task without designing specialized networks like ControlNet or IP-Adapter for a specific task.

**All you need to do is prepare the data and start training. You can break the limitations of previous models, allowing Omnigen to accomplish a variety of interesting tasks, even those that have never been done before.**


## Installation

```bash
git clone https://github.com/VectorSpaceLab/OmniGen.git
cd OmniGen
pip install -e .
```


## Full fine-tuning

### Fine-tuning command

```bash
accelerate launch \
    --num_processes=1 \
    --use_fsdp \
    --fsdp_offload_params false \
    --fsdp_sharding_strategy SHARD_GRAD_OP \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap Phi3DecoderLayer \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_forward_prefetch false \
    --fsdp_use_orig_params True \
    --fsdp_cpu_ram_efficient_loading false \
    --fsdp_sync_module_states True \
    train.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --json_file ./toy_data/toy_data.jsonl \
    --image_path ./toy_data/images \
    --batch_size_per_device 1 \
    --lr 2e-5 \
    --keep_raw_resolution \
    --max_image_size 1024 \
    --gradient_accumulation_steps 1 \
    --ckpt_every 50 \
    --epochs 200 \
    --log_every 1 \
    --results_dir ./results/toy_finetune
```

Some important arguments:
- `num_processes`: number of GPU to use for training
- `model_name_or_path`: path to the pretrained model
- `json_file`: path to the json file containing the training data, e.g., ./toy_data/toy_data.jsonl
- `image_path`: path to the image folder, e.g., ./toy_data/images
- `batch_size_per_device`: batch size per device
- `lr`: learning rate
- `keep_raw_resolution`: whether to keep the original resolution of the image, if not, all images will be resized to (max_image_size, max_image_size)
- `max_image_size`: max image size
- `gradient_accumulation_steps`: number of steps to accumulate gradients
- `ckpt_every`: number of steps to save checkpoint
- `epochs`: number of epochs
- `log_every`: number of steps to log
- `results_dir`: path to the results folder

The data format of json_file is as follows:
```
{
    "instruction": str, 
    "input_images": [str, str, ...], 
    "output_images": str
}
```
You can see a toy example in `./toy_data/toy_data.jsonl`.

If an OOM(Out of Memory) issue occurs, you can try to decrease the `batch_size_per_device` or `max_image_size`. You can also try to use LoRA instead of full fine-tuning.
## Quick Experiments

## Quick eval 
### evaluation on visual cloze
```bash
CUDA_VISIBLE_DEVICES=0 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 0
CUDA_VISIBLE_DEVICES=0 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 1
CUDA_VISIBLE_DEVICES=0 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 2
CUDA_VISIBLE_DEVICES=0 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 3
CUDA_VISIBLE_DEVICES=1 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 5
CUDA_VISIBLE_DEVICES=1 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 6
CUDA_VISIBLE_DEVICES=1 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 7
CUDA_VISIBLE_DEVICES=1 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 8
CUDA_VISIBLE_DEVICES=2 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 9
CUDA_VISIBLE_DEVICES=2 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 10
CUDA_VISIBLE_DEVICES=2 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 11
CUDA_VISIBLE_DEVICES=2 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 12
CUDA_VISIBLE_DEVICES=2 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 13
CUDA_VISIBLE_DEVICES=3 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 14
CUDA_VISIBLE_DEVICES=3 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 15
CUDA_VISIBLE_DEVICES=3 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 16
CUDA_VISIBLE_DEVICES=3 python evaluations/eval_visualcloze.py --is_parainj --adapter_path /nvme-data/Komal/documents/results/VisualCloze/qr_new/checkpoints/0002600/ --chunk 4
```

### comprehensive experimental eval
```bash
CUDA_VISIBLE_DEVICES=3 python eval_instruction.py --output_dir /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/comprehensive_experiments/inj/blip --adapter_path /nvme-data/Komal/documents/results/DREAMBOOT/inj/blip2full/qr/checkpoints/ --is_parainj 


CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --master_port 29507 generate_images.py --method dreambooth_deft_GFR4_sdxl --use_default_params True --db_or_ti_output_dir /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/submodules/dreambench_plus/work_dirs/dreambench_plus/dreambooth_deft_GFR4_sdxl_progress
```
## Training start
```bash
accelerate launch --num_processes=1 train.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --batch_size_per_device 2 \
    --condition_dropout_prob 0.01 \
    --lr 1e-3 \
    --use_lora \
    --lora_rank 8 \
    --json_file ./toy_data/toy_subject_data.jsonl \
    --image_path ./toy_data/images \
    --max_input_length_limit 18000 \
    --keep_raw_resolution \
    --max_image_size 1024 \
    --gradient_accumulation_steps 1 \
    --ckpt_every 10 \
    --epochs 200 \
    --log_every 1 \
    --results_dir ./results/toy_finetune_lora
```
```bash
## Sks Dog

    accelerate launch --num_processes=1 train.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --batch_size_per_device 2 \
    --condition_dropout_prob 0.01 \
    --lr 1e-3 \
    --use_para \
    --lora_rank 8 \
    --json_file ./toy_data/toy_subject_data.jsonl \
    --image_path ./toy_data/images \
    --max_input_length_limit 18000 \
    --keep_raw_resolution \
    --max_image_size 1024 \
    --gradient_accumulation_steps 1 \
    --ckpt_every 10 \
    --epochs 200 \
    --log_every 1 \
    --results_dir ./results/toy_finetune_lora
```
```bash
## DEFT Sks Dog

    accelerate launch --num_processes=1 train.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --batch_size_per_device 2 \
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
    --ckpt_every 10 \
    --epochs 200 \
    --log_every 1 \
    --decomposition_method None \
    --results_dir /nvme-data/Komal/documents/results/Abalation/No_decompo
```
```bash
## Training PARA
accelerate launch --num_processes=1 train_para.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 2     --condition_dropout_prob 0.01     --lr 1e-3     --use_para     --lora_rank 32     --json_file ./toy_data/toy_subject_data.jsonl     --image_path ./toy_data/images     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024     --gradient_accumulation_steps 1     --ckpt_every 100     --epochs 200     --log_every 1     --results_dir ./results/leusure_zone/

## Training svd para
accelerate launch --num_processes=1 train_para.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 2     --condition_dropout_prob 0.01     --lr 1e-3     --use_para     --lora_rank 32  --use_svd   --json_file ./toy_data/schene_obj_subject_data.jsonl     --image_path ./toy_data/table_scheen/     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024     --gradient_accumulation_steps 4     --ckpt_every 1000     --epochs 2000     --log_every 1     --results_dir /nvme-data/Komal/documents/results/schene_svd
### Leisure zone
accelerate launch --num_processes=2 train_para.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 2     --condition_dropout_prob 0.01     --lr 1e-3     --use_para     --lora_rank 8     --json_file ./toy_data/leisure_zone_sceens.jsonl     --image_path /nvme-data/Komal/documents/omni_datasets/InsDet-FULL/Scenes/easy/leisure_zone_001     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024     --gradient_accumulation_steps 1     --ckpt_every 100     --epochs 200     --log_every 1     --results_dir ./results/leusure_zone/

### Objects
accelerate launch --num_processes=2 train_para.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 8     --condition_dropout_prob 0.01     --lr 1e-3     --use_para     --lora_rank 8     --json_file ./toy_data/leisure_zone_sceens.jsonl     --image_path /nvme-data/Komal/documents/omni_datasets/InsDet-FULL/Scenes/easy/leisure_zone_001     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024     --gradient_accumulation_steps 1     --ckpt_every 100     --epochs 200     --log_every 1     --results_dir ./results/leusure_zone/

## SFM guided
accelerate launch --num_processes=1 train_para.py    --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 1     --condition_dropout_prob 0.01     --lr 1e-3     --use_para     --lora_rank 8  --json_file ./toy_data/sfm_scene.jsonl     --image_path /nvme-data/Komal/documents/omni_datasets/sfm/images     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 2048     --gradient_accumulation_steps 1     --ckpt_every 200     --epochs 400     --log_every 1     --results_dir ./results/sfm_based/

## SFM guided
accelerate launch --num_processes=1 train_para.py    --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 1     --condition_dropout_prob 0.01     --lr 1e-3     --use_para     --lora_rank 8  --json_file ./toy_data/sfm_scene.jsonl     --image_path /nvme-data/Komal/documents/omni_datasets/sfm/images     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 2048     --gradient_accumulation_steps 1     --ckpt_every 200     --epochs 400     --log_every 1     --results_dir /nvme-data/Komal/documents/results/sfm/para/
## DEFT
accelerate launch --num_processes=4 runner_aba.py    --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 2     --condition_dropout_prob 0.01     --lr 1e-3    --use_injection    --lora_rank 8  --json_file ./toy_data/sfm_scene.jsonl     --image_path /nvme-data/Komal/documents/omni_datasets/sfm/images     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024     --gradient_accumulation_steps 1     --ckpt_every 2000     --epochs 400     --log_every 1     --results_dir /nvme-data/Komal/documents/results/sfm/DEFT

## DREAMBOOT Objects
accelerate launch --num_processes=1 train_para.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 2     --condition_dropout_prob 0.01     --lr 1e-3     --use_para     --lora_rank 8    --json_file ./toy_data/toy_subject_data.jsonl     --image_path ./toy_data/images     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024     --gradient_accumulation_steps 1     --ckpt_every 100     --epochs 200     --log_every 1     --results_dir /nvme-data/Komal/documents/results/DREAMBOOT/dog6/lrmf/

## Integerating various methods
accelerate launch --num_processes=1 train_para.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 2     --condition_dropout_prob 0.01     --lr 1e-3     --use_para     --lora_rank 64   --json_file /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts/dreambooth_eval.jsonl     --image_path /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024     --gradient_accumulation_steps 1     --ckpt_every 2000     --epochs 2000     --log_every 1     --results_dir /nvme-data/Komal/documents/results/DREAMBOOT/full/qr/ --decomposition_method qr

## Injecting PEFT
accelerate launch --num_processes=1 train_para.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 2     --condition_dropout_prob 0.01     --lr 1e-3     --use_injection    --lora_rank 64   --json_file /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts/dreambooth_eval.jsonl     --image_path /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024     --gradient_accumulation_steps 1     --ckpt_every 2000     --epochs 2000     --log_every 1     --results_dir /nvme-data/Komal/documents/results/DREAMBOOT//inj/full/qr/ --decomposition_method qr

## General inj
accelerate launch --num_processes=1 train_para.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 2     --condition_dropout_prob 0.01     --lr 1e-3     --use_injection    --lora_rank 32   --json_file /nvme-data/Komal/documents/omni_datasets/InsDet-FULL/images_scenes_qwen.jsonl  --image_path /nvme-data/Komal/documents/omni_datasets/InsDet-FULL     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024 --gradient_accumulation_steps 1     --ckpt_every 2000     --epochs 2000     --log_every 1     --results_dir /nvme-data/Komal/documents/results/InsDet-FULL/SCN/inj/qwenfull/qr/ --decomposition_method qr

## General scene
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 train_para.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 2     --condition_dropout_prob 0.01     --lr 1e-3     --use_injection    --lora_rank 32   --json_file //nvme-data/Komal/documents/omni_datasets/InsDet-FULL/images_objectss_qwen.jsonl  --image_path /nvme-data/Komal/documents/omni_datasets/InsDet-FULL     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024 --gradient_accumulation_steps 1     --ckpt_every 2000     --epochs 2000     --log_every 1     --results_dir /nvme-data/Komal/documents/results/InsDet-FULL/OBJ/inj/qwen2full/qr/ --decomposition_method qr


python generate_models.py --method dreambooth_deftp_sdxl --start 0 --end 150
## VisualCloze

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes=2 train_para.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 8    --condition_dropout_prob 0.01     --lr 1e-3     --use_injection    --lora_rank 32   --json_file /media/mbzuaiser/SSD1/Komal/Graph200K/VisualCloze/train/omnijson.jsonl  --image_path ./ --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024 --gradient_accumulation_steps 1     --ckpt_every 2000     --epochs 2000     --log_every 1     --results_dir /nvme-data/Komal/documents/results/VisualCloze/qr/ --decomposition_method qr

## Visual cloze on LoRa
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes=2 train_para.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 8    --condition_dropout_prob 0.01     --lr 1e-3     --use_lora   --lora_rank 8   --json_file /media/mbzuaiser/SSD1/Komal/Graph200K/VisualCloze/trainomnijson_depth_only.jsonl  --image_path ./ --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024 --gradient_accumulation_steps 1     --ckpt_every 2000     --epochs 2000     --log_every 1     --results_dir /nvme-data/Komal/documents/results/VisualCloze/lora/ --decomposition_method qr

## Visual cloze on DEFT
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 train_para.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 2    --condition_dropout_prob 0.01     --lr 1e-3     --use_injection    --lora_rank 8   --json_file /media/mbzuaiser/SSD1/Komal/Graph200K/VisualCloze/trainomnijson_depth_only.jsonl  --image_path ./ --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024 --gradient_accumulation_steps 1     --ckpt_every 2000     --epochs 2000     --log_every 1     --results_dir /nvme-data/Komal/documents/results/VisualCloze/DEFT/no_decomp --decomposition_method qr


## All abalation
accelerate launch --num_processes=4 runner_aba.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 2     --condition_dropout_prob 0.01     --lr 1e-3     --use_injection    --lora_rank 64   --json_file /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts/dreambooth_eval.jsonl     --image_path /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024     --gradient_accumulation_steps 1     --ckpt_every 100     --epochs 2000     --log_every 1     --results_dir /nvme-data/Komal/documents/results/Abalation/ --decomposition_method qr

accelerate launch --num_processes=4 runner_aba.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 2     --condition_dropout_prob 0.01     --lr 1e-3     --use_para   --lora_rank 64   --json_file /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts/dreambooth_eval.jsonl     --image_path /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts     --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024     --gradient_accumulation_steps 1     --ckpt_every 2000     --epochs 2000     --log_every 1     --results_dir /nvme-data/Komal/documents/results/Abalation/ --decomposition_method wr


## Composition 

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes=2 --main_process_port 29502 runner_aba.py     --model_name_or_path Shitao/OmniGen-v1     --batch_size_per_device 2    --condition_dropout_prob 0.01     --lr 1e-3     --use_lorapara --use_injection  --lora_rank 64 --json_file /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts/dreambooth_eval.jsonl     --image_path /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts --max_input_length_limit 18000     --keep_raw_resolution     --max_image_size 1024 --gradient_accumulation_steps 1     --ckpt_every 2000     --epochs 2000     --log_every 1     --results_dir /nvme-data/Komal/documents/results/Abalation/lorapara --decomposition_method qr
```
## Evaluation all
```bash
CUDA_VISIBLE_DEVICES=3 python scaled_eval.py --output_dir /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/comprehensive_experiments/nmf --addaptor_path nvme-data/Komal/documents/results/DREAMBOOT/full/nmf/checkpoints/0008000/ --lora_rank 8 --lora_rank 8
```
## Quick generation
python eval_personalization.py --output_dir ./results//dreamboot/dog6/qr --decomposition_method 'qr'
## 2. Overview

OmniGen is a unified image generation model that can generate a wide range of images from multi-modal prompts. It is designed to be simple, flexible, and easy to use. We provide [inference code](#5-quick-start) so that everyone can explore more functionalities of OmniGen.

Existing image generation models often require loading several additional network modules (such as ControlNet, IP-Adapter, Reference-Net, etc.) and performing extra preprocessing steps (e.g., face detection, pose estimation, cropping, etc.) to generate a satisfactory image. However, **we believe that the future image generation paradigm should be more simple and flexible, that is, generating various images directly through arbitrarily multi-modal instructions without the need for additional plugins and operations, similar to how GPT works in language generation.** 

Due to the limited resources, OmniGen still has room for improvement. We will continue to optimize it, and hope it inspires more universal image-generation models. You can also easily fine-tune OmniGen without worrying about designing networks for specific tasks; you just need to prepare the corresponding data, and then run the [script](#6-finetune). Imagination is no longer limited; everyone can construct any image-generation task, and perhaps we can achieve very interesting, wonderful, and creative things.

If you have any questions, ideas, or interesting tasks you want OmniGen to accomplish, feel free to discuss with us: 2906698981@qq.com, wangyueze@tju.edu.cn, zhengliu1026@gmail.com. We welcome any feedback to help us improve the model.

# Eval
python Benchmark.py --gpu 2 --output_dir ./results/comprehensive_experiments --max_image_size 1024 --decomposition_method qr --run_all



CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=29504 eval_clip_and_dino.py --dir /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/eval/VisualCloze/depth/lora

## Repo QnA
## To use phi1.5 correctly, TypeError: cannot unpack non-iterable NoneType object
python -m pip install transformers==4.45.2
### To use Blip2 upgarde 
python -m pip install --upgrade git+https://github.com/huggingface/transformers.git



### Inference

The checkpoint can be found at `{results_dir}/checkpoints/*`. You can use the following command to load saved checkpoint:
```python
from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("checkpoint_path")  # e.g., ./results/toy_finetune/checkpoints/0000200
```





## LoRA fine-tuning
LoRA fine-tuning is a simple way to fine-tune OmniGen with less GPU memory. To use lora, you should add `--use_lora` and `--lora_rank` to the command.

```bash
accelerate launch \
    --num_processes=1 \
    train.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --batch_size_per_device 2 \
    --condition_dropout_prob 0.01 \
    --lr 3e-4 \
    --use_lora \
    --lora_rank 8 \
    --json_file ./toy_data/toy_data.jsonl \
    --image_path ./toy_data/images \
    --max_input_length_limit 18000 \
    --keep_raw_resolution \
    --max_image_size 1024 \
    --gradient_accumulation_steps 1 \
    --ckpt_every 50 \
    --epochs 100 \
    --log_every 1 \
    --results_dir ./results/toy_finetune_lora
```

### Inference

The checkpoint can be found at `{results_dir}/checkpoints/*`. You can use the following command to load checkpoint:
```python
from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
pipe.merge_lora("checkpoint_path")  # e.g., ./results/toy_finetune_lora/checkpoints/0000100
```


## A simple example

Here is an example for learning new concepts: "sks dog". We use five images of one dog from [dog-example](https://huggingface.co/datasets/diffusers/dog-example). 

The json file is `./toy_data/toy_subject_data.jsonl`, and the images have been saved in `./toy_data/images`.

```bash
accelerate launch \
    --num_processes=1 \
    train.py \
    --model_name_or_path Shitao/OmniGen-v1 \
    --batch_size_per_device 2 \
    --condition_dropout_prob 0.01 \
    --lr 1e-3 \
    --use_lora \
    --lora_rank 16 \
    --json_file ./toy_data/toy_subject_data.jsonl \
    --image_path ./toy_data/images \
    --max_input_length_limit 18000 \
    --keep_raw_resolution \
    --max_image_size 1024 \
    --gradient_accumulation_steps 1 \
    --ckpt_every 50 \
    --epochs 200 \
    --log_every 1 \
    --results_dir ./results/toy_finetune_lora
```

After training, you can use the following command to generate images:
```python
from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
pipe.merge_lora("checkpoint_path") # e.g., ./results/toy_finetune_lora/checkpoints/0000200

images = pipe(
    prompt="a photo of sks dog running in the snow", 
    height=1024, 
    width=1024, 
    guidance_scale=3
)
images[0].save("example_sks_dog_snow.png")
```
