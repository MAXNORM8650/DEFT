import os
import json
import torch
import argparse
import multiprocessing
from PIL import Image
from tqdm import tqdm
import megfile
from pathlib import Path
import sys
from functools import partial
import time
from peft import LoraConfig

# Add OmniGen to path if needed
sys.path.append("/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen")
from functions.image_utils import save_image

class LoraConfigExtended(LoraConfig):
    """Extended LoraConfig with decomposition method"""
    def __init__(self, decomposition_method=None, **kwargs):
        super().__init__(**kwargs)
        self.decomposition_method = decomposition_method


def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run OmniGen experiments across multiple GPUs')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='Comma-separated list of GPU device IDs')
    parser.add_argument('--processes_per_gpu', type=int, default=4, help='Number of processes per GPU')
    parser.add_argument('--output_dir', type=str, default='/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/eval/VisualCloze2/Omnigen', 
                        help='Output directory')
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to adapter weights')
    parser.add_argument('--is_para', action='store_true', help='Use PaRa adapter')
    parser.add_argument('--is_parainj', action='store_true', help='Use PaRa injection')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=16, help='LoRA alpha')
    parser.add_argument('--decomposition_method', type=str, default='qr', help='Decomposition method')
    parser.add_argument('--max_input_image_size', type=int, default=1024, help='Maximum input image size')
    parser.add_argument('--experiments_file', type=str, 
                        default="/media/mbzuaiser/SSD1/Komal/Graph200K/VisualCloze/testomnijson.jsonl",
                        help='JSONL file containing experiments')
    return parser.parse_args()


def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def last_three(path_input: str, path_output: str) -> str:
    """Extract path components to create a unique ID"""
    # Extract the last component of input and output paths
    input_last = os.path.basename(path_input).split('.')[0]  # '0_ref'
    output_last = os.path.basename(path_output).split('.')[0]  # '0_target'

    # Extract the directory names for input and output
    input_dir = os.path.dirname(path_input).split('/')[-1]  # 'ref'
    output_dir = os.path.dirname(path_output).split('/')[-1]  # 'target'

    # Combine them into the required structure
    new_id = f"{input_dir}/{input_last}/{output_dir}/{output_last}"
    
    return new_id


def load_multiline_jsonl(path):
    """Load a file that contains several pretty‑printed JSON objects one after another."""
    records = []
    buf, depth = [], 0  # depth = open braces – close braces

    with Path(path).open() as f:
        for line in f:
            # track how deep we are inside { ... }
            depth += line.count('{') - line.count('}')
            buf.append(line)

            if depth == 0 and buf:  # we just closed a complete object
                records.append(json.loads(''.join(buf)))
                buf.clear()  # start buffering the next object

    if buf:  # safety check – unbalanced braces
        raise ValueError("File ended before last object was closed")

    return records


import os
import json
from pathlib import Path

def load_experiments(file_path, output_dir):
    """Load experiments from JSONL file and check directory existence"""
    with Path(file_path).open() as f:
        experiments = []
        for line in f:
            if line.strip():
                exp = json.loads(line)
                # Check for task type
                # if exp['task_type'] not in ["depth", "canny"]:
                #     print(f"Task type {exp['task_type']} is not supported. Skipping this experiment.")
                #     continue  # Skip this experiment if task type is unsupported
                
                # Generate ID and check if target directory exists
                id = last_three(exp['input_images'][0], exp['output_image'])
                save_dir = output_dir
                method_dir = "omnigen-DEFT-32-2600"  # Default method directory
                target_dir = os.path.join(save_dir, method_dir, "tgt_image", id)

                if os.path.exists(target_dir):
                    print(f"Directory {target_dir} already exists. Skipping this experiment.")
                    continue  # Skip this experiment if the target directory already exists

                experiments.append(exp)
    return experiments

def process_experiment(exp, pipe, output_dir, max_input_image_size, process_id):
    """Process a single experiment"""
    try:
        print(f"[Process {process_id}] Running experiment: {exp['task_type']}")
        print(f"[Process {process_id}] Prompt: {exp['instruction'][:100]}...")
        id = last_three(exp['input_images'][0], exp['output_image'])
        save_dir = output_dir
        method_dir = "omnigen-DEFT-32-2600"  # Default method directory
        target_dir = os.path.join(save_dir, method_dir, "tgt_image", id)
        # if os.path.exists(target_dir):
        #     print(f"Directory {target_dir} already exists. Skipping this experiment.")
        #     return True
        # # if the task_type is "depth" or "canny" the run else skip
        # if exp['task_type'] not in ["depth", "canny"]:
        #     print(f"Task type {exp['task_type']} is not supported. Skipping this experiment.")
        #     return True
        # if exp['input_images']:
        #     print(f"[Process {process_id}] Input images:")
        #     for img_path in exp['input_images']:
        #         print(f"[Process {process_id}] - {img_path}")
        
        # Extract experiment ID from name
        # Generate images
        output = pipe(
            prompt=exp['instruction'],
            input_images=exp['input_images'] if 'input_images' in exp and exp['input_images'] else None,
            height=1024,
            width=1024,
            max_input_image_size=max_input_image_size,
            offload_model=False,  # Set to True if OOM issues occur
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
                save_image(cond_image, path=os.path.join(save_dir, method_dir, "src_image_input", id, f"{process_id}_{j}.jpg"))
            if expected_image:
                save_image(expected_image, path=os.path.join(save_dir, method_dir, "src_image_output", id, f"{process_id}_{j}.jpg"))            
            # Save generated image
            save_image(_output, path=os.path.join(save_dir, method_dir, "tgt_image", id, f"{process_id}_{j}.jpg"))
            
            # Save prompt
            prompt_gt = exp['instruction']
            with megfile.smart_open(os.path.join(save_dir, method_dir, "text", id, f"{process_id}_{j}.txt"), "w") as f:
                f.write(prompt_gt)
        
        return True
    except Exception as e:
        print(f"[Process {process_id}] Error processing experiment: {e}")
        return False


def worker_init(gpu_id, args):
    """Initialize worker with specific GPU"""
    print(f"Initializing worker on GPU {gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Import here to avoid importing in the main process
    from OmniGen import OmniGenPipeline
    from modules.utils import load_para_rank_adapter
    from modules.pepara import make_para_rank_adapter
    from modules.prepainj import add_knowledge_injection_methods
    
    # Load model
    print(f"[GPU {gpu_id}] Loading OmniGen model...")
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
    
    # Apply adapter according to arguments
    if args.is_para:
        transformer_lora_config = LoraConfigExtended(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["qkv_proj", "o_proj"],
            decomposition_method=args.decomposition_method
        )

        pipe.model = load_para_rank_adapter(pipe.model, args.adapter_path, transformer_lora_config)
        print(f"[GPU {gpu_id}] PaRa Gen Reduction applied successfully!")
    elif args.is_parainj:
        new_model = add_knowledge_injection_methods(pipe.model)
        pipe.model = new_model.load_knowledge_injection_adapter(new_model, args.adapter_path)
        print(f"[GPU {gpu_id}] Knowledge injection adapter loaded successfully!")
    
    print(f"[GPU {gpu_id}] Model initialized successfully!")
    return pipe


def process_batch(batch_experiments, gpu_id, args, process_id):
    """Process a batch of experiments on a specific GPU"""
    pipe = worker_init(gpu_id, args)
    
    results = []
    for exp in batch_experiments:
        success = process_experiment(exp, pipe, args.output_dir, args.max_input_image_size, process_id)
        results.append(success)
    
    return results


def main():
    args = setup_args()
    
    # Ensure output directory exists
    ensure_output_dir(args.output_dir)
    print(f"Outputs will be saved to: {args.output_dir}")
    
    # Load experiments
    print(f"Loading experiments from {args.experiments_file}...")
    experiments = load_experiments(args.experiments_file, args.output_dir)
    print(f"Loaded {len(experiments)} experiments")
    
    # Parse GPU IDs
    gpu_ids = [int(gpu_id) for gpu_id in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    
    total_processes = num_gpus * args.processes_per_gpu
    print(f"Total processes: {total_processes}")
    
    # Distribute experiments among processes
    batches = []
    experiments_per_process = len(experiments) // total_processes
    remainder = len(experiments) % total_processes
    
    start_idx = 0
    for i in range(total_processes):
        # Add one extra experiment to some processes if the division isn't even
        extra = 1 if i < remainder else 0
        batch_size = experiments_per_process + extra
        end_idx = start_idx + batch_size
        
        batches.append(experiments[start_idx:end_idx])
        start_idx = end_idx
    
    # Assign GPU ID for each process
    process_gpu_map = {}
    for i in range(total_processes):
        process_gpu_map[i] = gpu_ids[i % num_gpus]
    
    # Create process pool
    print("Starting multiprocessing pool...")
    with multiprocessing.Pool(processes=total_processes) as pool:
        # Prepare tasks for each process
        tasks = []
        for process_id, batch in enumerate(batches):
            gpu_id = process_gpu_map[process_id]
            tasks.append((batch, gpu_id, args, process_id))
        
        # Execute tasks in parallel
        results = pool.starmap(process_batch, tasks)
    
    print("\n=== All experiments completed successfully! ===")
    print(f"Results are saved in {args.output_dir}")


if __name__ == "__main__":
    # Required for Windows - not needed for Linux/MacOS
    multiprocessing.freeze_support()
    main()