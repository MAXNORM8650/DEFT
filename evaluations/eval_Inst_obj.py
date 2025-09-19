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
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to adapter weights')
    parser.add_argument('--is_para', action='store_true', help='Use PaRa adapter')
    parser.add_argument('--is_parainj', action='store_true', help='Use PaRa injection')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=16, help='LoRA alpha')
    parser.add_argument('--decomposition_method', type=str, default='qr', help='Decomposition method')
    parser.add_argument('--max_input_image_size', type=int, default=1024, help='Maximum input image size')
    return parser.parse_args()


def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

import json
from pathlib import Path

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
        print(f"\nRunning experiment: {exp['name']}")
        print(f"Prompt: {exp['prompt'][:100]}...")
        
        if exp['input_images']:
            print("Input images:")
            for img_path in exp['input_images']:
                print(f"- {img_path}")
        
        # Extract experiment ID from expt
        id = generate_logical_id(exp)
        id+=f"_{i}"
        method_dir = "omnigen"  # Default method directory
        save_dir = output_dir
        
        # Generate images
        output = pipe(
            prompt=exp['prompt'],
            input_images=exp['input_images'] if 'input_images' in exp and exp['input_images'] else None,
            height=1024,
            width=1024,
            max_input_image_size=max_input_image_size,
            offload_model=False,  # Set to True if OOM issues occur
            **exp['params']
        )
        
        # Get source image if available
        cond_image = None
        if exp['input_images'] and len(exp['input_images']) > 0:
            cond_image = Image.open(exp['input_images'][0])
        
        # Save results in the specified format
        for j, _output in enumerate(output):
            # Create directories
            os.makedirs(os.path.join(save_dir, method_dir, "src_image", id), exist_ok=True)
            os.makedirs(os.path.join(save_dir, method_dir, "tgt_image", id), exist_ok=True)
            os.makedirs(os.path.join(save_dir, method_dir, "text", id), exist_ok=True)
            
            # Save source image if available
            if cond_image:
                save_image(cond_image, path=os.path.join(save_dir, method_dir, "src_image", id, f"{i}_{j}.jpg"))
            
            # Save generated image
            save_image(_output, path=os.path.join(save_dir, method_dir, "tgt_image", id, f"{i}_{j}.jpg"))
            
            # Save prompt
            prompt_gt = exp['prompt']
            with megfile.smart_open(os.path.join(save_dir, method_dir, "text", id, f"{i}_{j}.txt"), "w") as f:
                f.write(prompt_gt)
                
        pbar.update(1)
    
    pbar.close()
import re
from pathlib import Path

def get_image_id(image_path):
    p = Path(image_path)
    parts = p.parts
    object_dir = parts[-3]  # e.g., "075_mouse_thinkpad"
    img_name = Path(parts[-1]).stem  # e.g., "001"
    return f"{object_dir}_{img_name}"

def get_prompt_keywords(prompt, max_keywords=3):
    # Remove image tag and lowercase
    prompt = prompt.lower().replace("<img><|image_1|></img>", "")
    # Extract meaningful words (nouns/adjectives)
    words = re.findall(r'\b[a-z]+\b', prompt)
    # Choose some key words (you can improve this with NLP later)
    important = [w for w in words if w not in {"the", "a", "is", "on", "with", "in", "of", "and", "as", "by"}]
    return "_".join(important[:max_keywords])

def generate_logical_id(exp):
    image_id = get_image_id(exp['input_images'][0])
    prompt_part = get_prompt_keywords(exp['prompt'])
    return f"{image_id}__{prompt_part}"

def main():
    args = setup_args()
    from modules.utils import load_para_rank_adapter, add_para_rank_methods
    from modules.pepara import make_para_rank_adapter
    from modules.prepainj import add_knowledge_injection_methods
    from peft import LoraConfig
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU: {args.gpu}")
    
    # Create output directories
    ensure_output_dir(args.output_dir)
    print(f"Outputs will be saved to: {args.output_dir}")
    # Load experiments from JSON
    print("Loading experiments from JSON...")
    experiments_file = "/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/INS-OBJ-EVAL.jsonl"

    from pathlib import Path

    with Path("/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/eval_l3q_obj.json").open() as f:
        cfg = json.load(f)

    print(cfg["experiments"][0]["name"])
# → consistent_object_different_scenes_1

    experiments_data = cfg
    experiments = experiments_data.get("experiments", [])
    print(f"Loaded {len(experiments)} experiments")    
    # Import and load model
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
        new_model = add_knowledge_injection_methods(pipe.model)
        pipe.model = new_model.load_knowledge_injection_adapter(new_model, adapter_path)
        print("Knowledge injection adapter loaded successfully!")
    
    # Run experiments
    run_experiments(pipe, experiments, args.output_dir, args.max_input_image_size)

    print("\n=== All experiments completed successfully! ===")
    print(f"Results are saved in {args.output_dir}")


if __name__ == "__main__":
    main()