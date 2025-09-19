import os
import argparse
import json
import sys
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
import megfile
from pathlib import Path
import time
import signal
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Add the OmniGen path to the system path
sys.path.append("/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen")
from functions.image_utils import save_image

# Import necessary classes from your script
from peft import LoraConfig

class LoraConfigExtended(LoraConfig):
    """Extended LoraConfig with decomposition method"""
    def __init__(self, decomposition_method=None, **kwargs):
        super().__init__(**kwargs)
        self.decomposition_method = decomposition_method


def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run OmniGen experiments in parallel')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU device IDs')
    parser.add_argument('--output_dir', type=str, default='/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/eval/VisualCloze2', help='Output directory')
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to adapter weights')
    parser.add_argument('--is_para', action='store_true', help='Use PaRa adapter')
    parser.add_argument('--is_parainj', action='store_true', help='Use PaRa injection')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=16, help='LoRA alpha')
    parser.add_argument('--decomposition_method', type=str, default='qr', help='Decomposition method')
    parser.add_argument('--max_input_image_size', type=int, default=1024, help='Maximum input image size')
    parser.add_argument('--experiments_per_gpu', type=int, default=4, help='Number of experiments to run in parallel per GPU')
    parser.add_argument('--experiment_file', type=str, default='/media/mbzuaiser/SSD1/Komal/Graph200K/VisualCloze/testomnijson.jsonl', help='Path to experiments JSONL file')
    return parser.parse_args()


def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def last_three(path_input: str, path_output: str) -> str:
    # Extract the last two components of input and output paths
    input_last = os.path.basename(path_input).split('.')[0]  # '0_ref'
    output_last = os.path.basename(path_output).split('.')[0]  # '0_target'

    # Extract the directory names for input and output
    input_dir = os.path.dirname(path_input).split('/')[-1]  # 'ref'
    output_dir = os.path.dirname(path_output).split('/')[-1]  # 'target'

    # Combine them into the required structure
    new_id = f"{input_dir}/{input_last}/{output_dir}/{output_last}"
    return new_id


def load_experiments(experiment_file, output_dir):
    """Load experiments from a JSONL file and filter them"""
    valid_experiments = []
    method_dir = "omnigen-DEFT-32-2600"  # Default method directory
    save_dir = output_dir
    
    with Path(experiment_file).open() as f:
        for line in f:
            if line.strip():
                exp = json.loads(line)
                # Check for task type
                if exp['task_type'] not in ["depth", "canny"]:
                    continue  # Skip this experiment if task type is unsupported
                
                # Generate ID and check if target directory exists
                id = last_three(exp['input_images'][0], exp['output_image'])
                target_dir = os.path.join(save_dir, method_dir, "tgt_image", id)

                if os.path.exists(target_dir):
                    continue  # Skip this experiment if the target directory already exists

                valid_experiments.append(exp)
    
    return valid_experiments


def run_single_experiment(pipe, exp, output_dir, max_input_image_size, exp_index):
    """Run a single experiment and save results"""
    try:
        print(f"\nRunning experiment {exp_index}: {exp['task_type']}")
        print(f"Prompt: {exp['instruction'][:100]}...")
        
        if exp['input_images']:
            print(f"Input image: {exp['input_images'][0]}")
        
        # Extract experiment ID from name
        id = last_three(exp['input_images'][0], exp['output_image'])
        method_dir = "omnigen-DEFT-32-2600"  # Default method directory
        save_dir = output_dir
        
        # Create target directories first to prevent race conditions
        os.makedirs(os.path.join(save_dir, method_dir, "src_image_input", id), exist_ok=True)
        os.makedirs(os.path.join(save_dir, method_dir, "src_image_output", id), exist_ok=True)
        os.makedirs(os.path.join(save_dir, method_dir, "tgt_image", id), exist_ok=True)
        os.makedirs(os.path.join(save_dir, method_dir, "text", id), exist_ok=True)
        
        # Generate images with proper error handling
        output = None
        try:
            output = pipe(
                prompt=exp['instruction'],
                input_images=exp['input_images'] if 'input_images' in exp and exp['input_images'] else None,
                height=1024,
                width=1024,
                max_input_image_size=max_input_image_size,
                offload_model=False,  # Set to True if OOM issues occur
            )
        except Exception as e:
            print(f"Error generating image for experiment {exp_index}: {str(e)}")
            traceback.print_exc()
            return exp_index, f"ERROR-{id}"
        
        if not output:
            print(f"No output generated for experiment {exp_index}")
            return exp_index, f"EMPTY-{id}"
        
        # Get source image if available
        cond_image = None
        if exp['input_images'] and len(exp['input_images']) > 0:
            try:
                cond_image = Image.open(exp['input_images'][0])
            except Exception as e:
                print(f"Error opening input image for experiment {exp_index}: {str(e)}")
        
        expected_image = None
        if exp['output_image']:
            try:
                expected_image = Image.open(exp['output_image'])
            except Exception as e:
                print(f"Error opening output image for experiment {exp_index}: {str(e)}")

        # Save results in the specified format
        for j, _output in enumerate(output):
            try:
                # Save source image if available
                if cond_image:
                    save_image(cond_image, path=os.path.join(save_dir, method_dir, "src_image_input", id, f"{exp_index}_{j}.jpg"))
                if expected_image:
                    save_image(expected_image, path=os.path.join(save_dir, method_dir, "src_image_output", id, f"{exp_index}_{j}.jpg"))            
                
                # Save generated image
                save_image(_output, path=os.path.join(save_dir, method_dir, "tgt_image", id, f"{exp_index}_{j}.jpg"))
                
                # Save prompt
                prompt_gt = exp['instruction']
                with megfile.smart_open(os.path.join(save_dir, method_dir, "text", id, f"{exp_index}_{j}.txt"), "w") as f:
                    f.write(prompt_gt)
            except Exception as e:
                print(f"Error saving results for experiment {exp_index}, output {j}: {str(e)}")
                traceback.print_exc()
        
        return exp_index, id
    
    except Exception as e:
        print(f"Unexpected error in experiment {exp_index}: {str(e)}")
        traceback.print_exc()
        return exp_index, "ERROR"


def run_gpu_experiments(gpu_id, experiments, args):
    """Run multiple experiments in parallel on a single GPU"""
    try:
        # Set the CUDA device for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)  # Within this process, the GPU is always device 0
        print(f"Process for GPU {gpu_id} started, handling {len(experiments)} experiments")
        
        # Import and load model - do this inside the process to ensure proper GPU allocation
        from OmniGen import OmniGenPipeline
        from modules.utils import load_para_rank_adapter
        from modules.pepara import make_para_rank_adapter
        from modules.prepainj import add_knowledge_injection_methods
        
        # Load the model
        print(f"GPU {gpu_id}: Loading OmniGen model...")
        pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
        
        # Apply adapters
        if args.is_para:
            transformer_lora_config = LoraConfigExtended(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["qkv_proj", "o_proj"],
                decomposition_method=args.decomposition_method
            )
            pipe.model = load_para_rank_adapter(pipe.model, args.adapter_path, transformer_lora_config)
            print(f"GPU {gpu_id}: PaRa Gen Reduction applied successfully!")
        elif args.is_parainj:
            new_model = add_knowledge_injection_methods(pipe.model)
            pipe.model = new_model.load_knowledge_injection_adapter(new_model, args.adapter_path)
            print(f"GPU {gpu_id}: Knowledge injection adapter loaded successfully!")
        
        # Move model to GPU
        # pipe
        
        # Process experiments sequentially or in smaller batches to avoid memory issues
        max_concurrent = min(args.experiments_per_gpu, 4)  # Limit max concurrent to prevent resource issues
        
        completed = 0
        with tqdm(total=len(experiments), desc=f"GPU {gpu_id} experiments") as pbar:
            for i in range(0, len(experiments), max_concurrent):
                batch = experiments[i:i+max_concurrent]
                
                # Use ThreadPoolExecutor for parallelism within the GPU
                with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    futures = []
                    
                    # Submit batch of experiments to the thread pool
                    for j, exp in enumerate(batch):
                        future = executor.submit(
                            run_single_experiment, 
                            pipe, exp, args.output_dir, 
                            args.max_input_image_size, 
                            i+j
                        )
                        futures.append(future)
                    
                    # Process results as they complete
                    for future in as_completed(futures):
                        try:
                            exp_index, exp_id = future.result()
                            print(f"GPU {gpu_id}: Completed experiment {exp_index} with ID {exp_id}")
                            completed += 1
                            pbar.update(1)
                        except Exception as e:
                            print(f"GPU {gpu_id}: Error in experiment: {str(e)}")
                            traceback.print_exc()
                            pbar.update(1)
                
                # Force garbage collection between batches
                gc.collect()
                torch.cuda.empty_cache()
        
        print(f"GPU {gpu_id}: All assigned experiments completed ({completed}/{len(experiments)})")
        
        # Clean up resources
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"ERROR in GPU {gpu_id} process: {str(e)}")
        traceback.print_exc()
    
    finally:
        # Ensure we clean up resources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    # Parse arguments
    args = setup_args()
    
    # Create output directories
    ensure_output_dir(args.output_dir)
    print(f"Outputs will be saved to: {args.output_dir}")
    
    # Get available GPUs
    gpu_ids = [id.strip() for id in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    
    # Load and filter experiments
    print(f"Loading experiments from {args.experiment_file}...")
    experiments = load_experiments(args.experiment_file, args.output_dir)
    print(f"Loaded {len(experiments)} valid experiments to process")
    
    if not experiments:
        print("No valid experiments to run. Exiting.")
        return
    
    # Divide experiments among GPUs
    gpu_experiments = []
    experiments_per_gpu = (len(experiments) + num_gpus - 1) // num_gpus  # Ceiling division
    
    for i in range(num_gpus):
        start_idx = i * experiments_per_gpu
        end_idx = min(start_idx + experiments_per_gpu, len(experiments))
        gpu_experiments.append(experiments[start_idx:end_idx])
        print(f"GPU {gpu_ids[i]} will process {len(gpu_experiments[-1])} experiments")
    
    try:
        # Initialize multiprocessing with proper start method
        mp.set_start_method('spawn', force=True)
        
        # Start processes for each GPU
        processes = []
        for i, (gpu_id, exps) in enumerate(zip(gpu_ids, gpu_experiments)):
            if exps:  # Only start a process if there are experiments to run
                p = mp.Process(target=run_gpu_experiments, args=(gpu_id, exps, args))
                p.daemon = True  # Set as daemon to ensure it exits when main process exits
                processes.append(p)
                p.start()
                # Small delay to avoid race conditions during startup
                time.sleep(5)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Terminating processes...")
        # Handle clean termination
        for p in processes:
            if p.is_alive():
                p.terminate()
        print("Processes terminated.")
        return
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        traceback.print_exc()
        # Clean up processes
        for p in processes:
            if p.is_alive():
                p.terminate()
        return
    
    print("\n=== All experiments completed successfully! ===")
    print(f"Results are saved in {args.output_dir}")


if __name__ == "__main__":
    main()