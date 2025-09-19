import os
from PIL import Image
from tqdm import tqdm
import megfile
import argparse
import json
import sys
sys.path.append("/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen")
from functions.image_utils import save_image
import os
import torch
import json
import megfile
from PIL import Image
from tqdm import tqdm
from peft import LoraConfig
from pathlib import Path

class LoraConfigExtended(LoraConfig):
    """Extended LoraConfig with decomposition method"""
    def __init__(self, decomposition_method=None, **kwargs):
        super().__init__(**kwargs)
        self.decomposition_method = decomposition_method


def setup_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Run OmniGen experiments')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID')
    parser.add_argument('--output_dir', type=str, default='/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/eval/VisualCloze2', help='Output directory')
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to adapter weights')
    parser.add_argument('--is_para', action='store_true', help='Use PaRa adapter')
    parser.add_argument('--is_parainj', action='store_true', help='Use PaRa injection')
    parser.add_argument('--is_lora', action='store_true', help='Use LoRA adapter')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=16, help='LoRA alpha')
    parser.add_argument('--decomposition_method', type=str, default='qr', help='Decomposition method')
    parser.add_argument('--max_input_image_size', type=int, default=1024, help='Maximum input image size')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk number for parallel processing')
    return parser.parse_args()


def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

import json
from pathlib import Path
def last_three(path_input: str, path_output: str) -> str:
    # Extract the last two components of input and output paths
    input_last = os.path.basename(path_input).split('.')[0]  # '0_ref'
    output_last = os.path.basename(path_output).split('.')[0]  # '0_target'

    # Extract the directory names for input and output
    input_dir = os.path.dirname(path_input).split('/')[-1]  # 'ref'
    output_dir = os.path.dirname(path_output).split('/')[-1]  # 'target'

    # Combine them into the required structure
    new_id = f"{input_dir}/{input_last}/{output_dir}/{output_last}"
    
    print(new_id)
    return new_id

def load_multiline_jsonl(path):
    """Load a file that contains several pretty‑printed JSON objects one after another."""
    records = []
    buf, depth = [], 0                      # depth = open braces – close braces

    with Path(path).open() as f:
        for line in f:
            # track how deep we are inside { … }
            depth += line.count('{') - line.count('}')
            buf.append(line)

            if depth == 0 and buf:          # we just closed a complete object
                records.append(json.loads(''.join(buf)))
                buf.clear()                 # start buffering the next object

    if buf:                                 # safety check – unbalanced braces
        raise ValueError("File ended before last object was closed")

    return records

def run_experiments(pipe, experiments, output_dir, max_input_image_size):
    """Helper function to run experiments"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a progress bar for all experiments
    total_experiments = len(experiments)
    pbar = tqdm(total=total_experiments, desc="Running experiments")
    
    for i, exp in enumerate(experiments):
        print(f"\nRunning experiment: {exp['task_type']}")
        print(f"Prompt: {exp['instruction'][:100]}...")
        
        if exp['input_images']:
            print("Input images:")
            for img_path in exp['input_images']:
                print(f"- {img_path}")
        
        # Extract experiment ID from name
        id = last_three(exp['input_images'][0], exp['output_image'])
        method_dir = "omnigen-lora-08-4000-depth"  # Default method directory
        save_dir = output_dir
        output = pipe(
            prompt=exp['instruction'],
            input_images=exp['input_images'] if 'input_images' in exp and exp['input_images'] else None,
            height=1024,
            width=1024,
            max_input_image_size=max_input_image_size,
            offload_model=False,  # Set to True if OOM issues occur
            # **exp['params']
        )
        
        # Get source image if available
        cond_image = None
        if exp['input_images'] and len(exp['input_images']) > 0:
            cond_image = Image.open(exp['input_images'][0])

        expected_image = None
        if exp['output_image']:
            expected_image = Image.open(exp['output_image'])

        # Save results in the specified format
        for j, _output in enumerate(output):
            # Create directories
            os.makedirs(os.path.join(save_dir, method_dir, "src_image_input", id), exist_ok=True)
            os.makedirs(os.path.join(save_dir, method_dir, "src_image_output", id), exist_ok=True)
            os.makedirs(os.path.join(save_dir, method_dir, "tgt_image", id), exist_ok=True)
            os.makedirs(os.path.join(save_dir, method_dir, "text", id), exist_ok=True)
            
            # Save source image if available
            if cond_image:
                save_image(cond_image, path=os.path.join(save_dir, method_dir, "src_image_input", id, f"{i}_{j}.jpg"))
            if expected_image:
                save_image(expected_image, path=os.path.join(save_dir, method_dir, "src_image_output", id, f"{i}_{j}.jpg"))            
            # Save generated image
            save_image(_output, path=os.path.join(save_dir, method_dir, "tgt_image", id, f"{i}_{j}.jpg"))
            
            # Save prompt
            prompt_gt = exp['instruction']
            with megfile.smart_open(os.path.join(save_dir, method_dir, "text", id, f"{i}_{j}.txt"), "w") as f:
                f.write(prompt_gt)
        
        pbar.update(1)
    
    pbar.close()


def main():
    args = setup_args()
    from modules.utils import load_para_rank_adapter, add_para_rank_methods
    from modules.pepara import make_para_rank_adapter
    from modules.deft import add_knowledge_injection_methods
    from peft import LoraConfig
    
    
    # Create output directories
    ensure_output_dir(args.output_dir)
    print(f"Outputs will be saved to: {args.output_dir}")
    output_dir = args.output_dir
    # Load experiments from JSON
    print("Loading experiments from JSON...")

    from pathlib import Path

    with Path("/media/mbzuaiser/SSD1/Komal/Graph200K/VisualCloze/testomnijson.jsonl").open() as f:
        experiments_data = []
        for line in f:
            if line.strip():
                exp = json.loads(line)
                # Check for task type
                if exp['task_type'] not in ["depth"]: #["depth", "canny"]
                    print(f"Task type {exp['task_type']} is not supported. Skipping this experiment.")
                    continue  # Skip this experiment if task type is unsupported
                
                # Generate ID and check if target directory exists
                id = last_three(exp['input_images'][0], exp['output_image'])
                save_dir = output_dir
                method_dir = "omnigen-lora-08-4000-depth"  # Default method directory
                target_dir = os.path.join(save_dir, method_dir, "tgt_image", id)

                if os.path.exists(target_dir):
                    print(f"Directory {target_dir} already exists. Skipping this experiment.")
                    continue  # Skip this experiment if the target directory already exists

                experiments_data.append(exp)
    
    # experiments = experiments_data#.get("experiments", [])

    print(f"Loaded {len(experiments_data)} experiments")    
    # chunk_size = len(experiments_data) // num_processes
    # chunks = [experiments_data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
    experiments = experiments_data #chunks[args.chunk]
    print("Loading OmniGen model...")
    from OmniGen import OmniGenPipeline

    adapter_path = args.adapter_path
    print(f"Loading adapter weights from: {adapter_path}")
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
    print("Model loaded successfully!")
    
    if args.is_para:
        transformer_lora_config = LoraConfigExtended(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["qkv_proj", "o_proj"],
            decomposition_method=args.decomposition_method
        )

        pipe.model = load_para_rank_adapter(pipe.model, adapter_path, transformer_lora_config)
        print("PaRa Gen Reduction applied successfully!")
    elif args.is_parainj:
        # breakpoint()
        new_model = add_knowledge_injection_methods(pipe.model)
        pipe.model = new_model.load_knowledge_injection_adapter(new_model, adapter_path)
        print("Knowledge injection adapter loaded successfully!")
    elif args.is_lora:
        print("Loading LoRA adapter...")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["qkv_proj", "o_proj"]
        )
        pipe.merge_lora(args.adapter_path)
        # pipe.model.load_adapter(args.adapter_path, lora_config)
    # Run experiments
    run_experiments(pipe, experiments, args.output_dir, args.max_input_image_size)

    print("\n=== All experiments completed successfully! ===")
    print(f"Results are saved in {args.output_dir}")


if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES=1 python evaluations/eval_visualcloze.py --output_dir /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/eval/VisualCloze/depth/lora --adapter_path /nvme-data/Komal/documents/results/VisualCloze/lora/depth/checkpoints/0004000/ --is_lora --lora_rank 8 --decomposition_method None