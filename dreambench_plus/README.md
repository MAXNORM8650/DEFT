# DEFT training and evaluation on [DreamBench++](https://github.com/yuangpeng/dreambench_plus)
## Overview

DreamBench++ establishes a fair and comprehensive benchmark for personalized image generation, addressing the limitations of existing evaluation frameworks through:

- **150 diverse reference images** across live subjects, objects, and artistic styles
- **1,350 carefully curated prompts** with varying complexity levels
- **Human-aligned evaluation metrics** powered by multimodal large language models
- **Automated evaluation pipeline** for consistent and scalable assessment

## Key Features

### Dataset Composition
- **Live Subjects**: 60 images (40 humans, 20 animals)  
- **Objects**: 60 images (everyday items, vehicles, etc.)
- **Artistic Styles**: 30 images (paintings, sketches, etc.)
- **Prompt Diversity**: 9 prompts per image across difficulty levels
  - 4 photorealistic style prompts
  - 3 non-photorealistic style prompts  
  - 2 complex/imaginative content prompts

### Evaluation Metrics
- **GPT-4V Score**: Human-aligned automated evaluation
- **DINO Score**: Identity preservation assessment
- **CLIP-I Score**: Image similarity measurement
- **CLIP-T Score**: Text-image alignment evaluation

## Quick Start

### Prerequisites

```bash
# Create and activate conda environment (Recommnaded)
conda create -n dreambench python=3.9
conda activate dreambench

# Clone repository
cd dreambench_plus

# Install package
pip install -e .
```

### Configuration

Before running experiments, configure your settings:

```bash
# Set data path in your environment or config
export DREAMBENCH_DATA_PATH="path/to/your/dreambench_plus_data"

# For GPT evaluation, set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

#### 1. Download Dataset

```bash
# Option 1: Google Drive
wget -O dreambench_plus_data.zip "https://drive.google.com/uc?id=17HNVYU5yvuHDC6VhesJsWsXo1UWy_CSs"
unzip dreambench_plus_data.zip

# Option 2: HuggingFace
git lfs clone https://huggingface.co/datasets/yuangpeng/dreambench_plus
```

#### 2. Preview Dataset

```bash
pip install streamlit
cd data
streamlit run preview.py
```

#### 3. Generate Images

For methods requiring fine-tuning (DreamBooth, Textual Inversion):

```bash
# Train models on all 150 samples
python generate_models.py --method dreambooth_sd --start 0 --end 150

# Or train on a subset for testing
python generate_models.py --method dreambooth_sd --start 0 --end 10
```

Generate images with your chosen method:

```bash
# Single GPU
python generate_images.py \
    --method blip_diffusion \
    --use_default_params True

# Multi-GPU (recommended for faster inference)
torchrun --nproc-per-node=8 generate_images.py \
    --method blip_diffusion \
    --use_default_params True

# With custom output directory
torchrun --nproc-per-node=8 generate_images.py \
    --method dreambooth_lora_sdxl \
    --db_or_ti_output_dir work_dirs/dreambench_plus/dreambooth_deft_sdxl \
    --use_default_params True
```

#### 4. Evaluate Results

**DINO and CLIP Scores:**
```bash
torchrun --nproc-per-node=8 eval_clip_and_dino.py \
    --dir samples/blip_diffusion_gs7_5_step100_seed42_torch_float16
```

**GPT-4V Evaluation:**
```bash
# Concept preservation
python eval_gpt.py \
    --method "DreamBooth LoRA SDXL" \
    --out_dir data_gpt_rating/concept_preservation_full/dreambooth_lora_sdxl \
    --category subject \
    --ablation_settings full

# Prompt following  
python eval_gpt.py \
    --method "DreamBooth LoRA SDXL" \
    --out_dir data_gpt_rating/prompt_following_full/dreambooth_lora_sdxl \
    --ablation_settings full
```

**Generate Final Benchmark Report:**
```bash
python benchmarking.py
```

### Batch Processing Scripts

Create efficient batch processing with our provided templates:

```bash
# Process multiple methods
bash scripts/batch_generate.sh

# Evaluate multiple experiments
bash scripts/batch_evaluate.sh
```

### Training Custom Models

For DEFT

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc-per-node=2 \
    --master_port=29501 \
    training_scripts/train_dreambooth_deft_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --instance_data_dir="$DREAMBENCH_DATA_PATH/samples/.../src_image/live_subject_animal_00_kitten" \
    --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
    --output_dir="work_dirs/dreambench_plus/deft_training" \
    --mixed_precision="fp16" \
    --instance_prompt="a photo of kitten" \
    --resolution=1024 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --max_train_steps=500 \
    --validation_epochs=50 \
    --seed=42
```

## Output Structure

Generated samples follow this organized structure:

```
samples/
└── {method_name}_{parameters}/
    ├── src_image/           # Reference images
    │   └── {category}_{id}_{name}/
    │       ├── 0_0.jpg     # Generated image for prompt 0, sample 0
    │       └── ...
    ├── text/               # Corresponding prompts
    │   └── {category}_{id}_{name}/
    │       ├── 0_0.txt
    │       └── ...
    ├── tgt_image/          # Target style references (if applicable)
    └── negative_prompt.txt # Global negative prompts
```
