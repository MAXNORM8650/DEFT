import json, os, itertools
from datetime import datetime
from tqdm import tqdm                         # just for a nicer progress bar
import os
import argparse
from PIL import Image
from pathlib import Path
from modules.utils import LoraConfigExtended

import os
import torch
from PIL import Image
import numpy as np

def setup_args():
    parser = argparse.ArgumentParser(description='Run OmniGen experiments')
    parser.add_argument('--gpu', type=str, default='3', help='GPU ID to use')
    parser.add_argument('--output_dir', type=str, default='./results/multi-personalization', help='Directory to save outputs')
    parser.add_argument('--max_image_size', type=int, default=1024, help='Maximum input image size')
    parser.add_argument('--decomposition_method', type=str, default='nmf', help='Decomposition method to use')
    return parser.parse_args()

def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)


import os
import json
import re
from collections import defaultdict

def run_comprehensive_evaluation(pipe, output_dir, max_input_image_size=1024):
    """Run comprehensive evaluation for diffusion models"""
    
    # Create evaluation directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Load concept information from JSON
    concepts = load_concept_info_from_json()
    
    # Run different evaluation experiments
    # run_identity_preservation_tests(pipe, output_dir, max_input_image_size, concepts)
    run_object_consistency_tests(pipe, output_dir, max_input_image_size)
    run_multi_object_composition_tests(pipe, output_dir, max_input_image_size)
    run_style_transfer_tests(pipe, output_dir, max_input_image_size)
    run_object_manipulation_tests(pipe, output_dir, max_input_image_size)
    
    # Compute metrics
    compute_evaluation_metrics(output_dir)

def load_concept_info_from_json(json_path="/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts/personalization_eval.jsonl"):
    """Extract concept information from the JSON file"""
    print(f"Loading concept information from {json_path}")
    
    concepts = defaultdict(list)
    concept_names = set()
    # breakpoint()
    try:
        with open(json_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            try:
                data = json.loads(line.strip())
                
                # Extract concept name from instruction
                instruction = data.get("instruction", "")
                match = re.search(r'sks ([\w\-]+)(?:\s+(.*))?', instruction)
                
                if match:
                    concept_name = match.group(1)
                    # Clean up concept name to handle variations
                    concept_name = concept_name.strip()
                    
                    # For compound concepts with additional words after (like "stuffed animal")
                    if " " in concept_name:
                        base_concept = concept_name.split(" ")[0]
                        concept_names.add(base_concept)
                    else:
                        concept_names.add(concept_name)
                        
                    # Store output image path
                    if "output_image" in data:
                        output_path = data["output_image"]
                        concept_dir = output_path.split('/')[0]  # Get the directory name
                        concepts[concept_dir].append(output_path)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON line: {line[:50]}...")
                continue
    
    except FileNotFoundError:
        print(f"Error: Could not find JSON file at {json_path}")
        return {}
    
    print(f"Found {len(concepts)} concepts: {', '.join(sorted(concepts.keys()))}")
    return concepts

def run_identity_preservation_tests(pipe, output_dir, max_input_image_size, concepts):
    """Run identity preservation experiments for all concepts"""
    print("\n=== Running Identity Preservation Experiments ===")
    
    # Create base output directory
    base_output_dir = os.path.join(output_dir, 'identity_preservation')
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Base path for concept images
    concept_root = "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts"
    
    # Person images for identity preservation tests
    person_images = {
        "man": "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/imgs/test_cases/young_trump.jpeg",
        "woman": "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/imgs/test_cases/woman.png",
        "two_people": "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/imgs/test_cases/two_man.jpg",
        "child": "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/imgs/test_cases/young_musk.jpg"
    }
    
    # Scenarios for identity preservation
    scenarios = [
        {
            "name": "simple_identity",
            "prompt_template": "A photo of a sks {concept} sitting on a table.",
            "input_images": [],
            "params": {
                "guidance_scale": 2.5,
                "seed": 42
            }
        },
        {
            "name": "person_interaction",
            "prompt_template": "A person is holding a sks {concept}. The sks {concept} is <img><|image_1|></img>.",
            "input_images_template": [concept_root + "/{concept_dir}/{sample_img}"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.6,
                "seed": 43
            }
        },
        {
            "name": "specific_person_interaction",
            "prompt_template": "A man is showing a sks {concept} to a woman. The man is the person in <img><|image_1|></img>. The sks {concept} is <img><|image_2|></img>.",
            "input_images_template": [person_images["man"], concept_root + "/{concept_dir}/{sample_img}"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.8,
                "seed": 44
            }
        },
        {
            "name": "outdoor_scene",
            "prompt_template": "A sks {concept} is in a park with trees and a lake in the background. The sks {concept} is <img><|image_1|></img>.",
            "input_images_template": [concept_root + "/{concept_dir}/{sample_img}"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.7,
                "seed": 45
            }
        },
        {
            "name": "indoor_scene",
            "prompt_template": "A sks {concept} is on a shelf in a cozy living room with a fireplace. The sks {concept} is <img><|image_1|></img>.",
            "input_images_template": [concept_root + "/{concept_dir}/{sample_img}"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.7,
                "seed": 46
            }
        },
        {
            "name": "unusual_context",
            "prompt_template": "A sks {concept} is floating in space with stars and planets in the background. The sks {concept} is <img><|image_1|></img>.",
            "input_images_template": [concept_root + "/{concept_dir}/{sample_img}"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.7,
                "seed": 47
            }
        },
        {
            "name": "multi_object_scene",
            "prompt_template": "A man is holding a sks {concept} while a woman is looking at it. The man is the person in <img><|image_1|></img>. The woman is the person in <img><|image_2|></img>. The sks {concept} is <img><|image_3|></img>.",
            "input_images_template": [person_images["man"], person_images["woman"], concept_root + "/{concept_dir}/{sample_img}"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.9,
                "seed": 48
            }
        }
    ]
    
    # Special prompts for animal concepts
    animal_concepts = ["cat", "cat2", "dog", "dog2", "dog3", "dog5", "dog6", "dog7", "dog8"]
    animal_scenarios = [
        {
            "name": "animal_behavior",
            "prompt_template": "A sks {concept} is playing with a ball. The sks {concept} is <img><|image_1|></img>.",
            "input_images_template": [concept_root + "/{concept_dir}/{sample_img}"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.7,
                "seed": 49
            }
        },
        {
            "name": "animal_expression",
            "prompt_template": "A sks {concept} is looking excited and happy. The sks {concept} is <img><|image_1|></img>.",
            "input_images_template": [concept_root + "/{concept_dir}/{sample_img}"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.7,
                "seed": 50
            }
        }
    ]
    
    # Special prompts for wearable concepts
    wearable_concepts = ["backpack", "backpack-dog", "fancy_boot", "colorful_sneaker", "shiny_sneaker", "pink_sunglasses"]
    wearable_scenarios = [
        {
            "name": "person_wearing",
            "prompt_template": "A person is wearing a sks {concept}. The person is <img><|image_1|></img>. The sks {concept} is <img><|image_2|></img>.",
            "input_images_template": [person_images["man"], concept_root + "/{concept_dir}/{sample_img}"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.8,
                "seed": 51
            }
        }
    ]
    
    # Special prompts for toy concepts
    toy_concepts = ["bear-plushie", "grey-sloth-plushie", "wolf-plushie", "duck_toy", "monster_toy", "poop-emoji", "rc-car", "red_cartoon", "robot_toy"]
    toy_scenarios = [
        {
            "name": "child_playing",
            "prompt_template": "A child is playing with a sks {concept}. The child is <img><|image_1|></img>. The sks {concept} is <img><|image_2|></img>.",
            "input_images_template": [person_images["child"], concept_root + "/{concept_dir}/{sample_img}"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.8,
                "seed": 52
            }
        }
    ]
    
    # Run experiments for each concept
    total_experiments = 0
    
    for concept_dir, image_paths in concepts.items():
        if not image_paths:
            continue
            
        # Get a representative sample image for this concept
        sample_img = os.path.basename(image_paths[0])
        
        # Extract the concept name from the directory
        concept_name = concept_dir.replace("_", "-")
        
        # Create output directory for this concept
        concept_output_dir = os.path.join(base_output_dir, concept_dir)
        os.makedirs(concept_output_dir, exist_ok=True)
        
        print(f"\nRunning identity preservation tests for concept: {concept_name}")
        
        # Determine which additional scenarios to use based on concept type
        additional_scenarios = []
        if any(animal in concept_dir for animal in ["cat", "dog"]):
            additional_scenarios.extend(animal_scenarios)
        if any(wearable in concept_dir for wearable in ["backpack", "boot", "sneaker", "sunglasses"]):
            additional_scenarios.extend(wearable_scenarios)
        if any(toy in concept_dir for toy in ["plushie", "toy", "emoji"]):
            additional_scenarios.extend(toy_scenarios)
        
        # Combine base scenarios with any additional ones
        all_scenarios = scenarios + additional_scenarios
        
        # Run each scenario for this concept
        for scenario in all_scenarios:
            # Format the prompt with the concept name
            prompt = scenario["prompt_template"].format(concept=concept_name)
            
            # Prepare input images
            input_images = []
            if "input_images_template" in scenario:
                for img_template in scenario["input_images_template"]:
                    input_image = img_template.format(concept_dir=concept_dir, sample_img=sample_img)
                    input_images.append(input_image)
            
            # Create experiment
            experiment = {
                "name": f"{concept_dir}_{scenario['name']}",
                "prompt": prompt,
                "input_images": input_images,
                "params": scenario["params"].copy()
            }
            
            # Run the experiment
            print(f"Running experiment: {experiment['name']}")
            print(f"Prompt: {experiment['prompt']}")
            if experiment['input_images']:
                print("Input images:")
                for img_path in experiment['input_images']:
                    print(f"- {img_path}")
            
            try:
                images = pipe(
                    prompt=experiment['prompt'],
                    input_images=experiment['input_images'] if experiment['input_images'] else None,
                    height=1024,
                    width=1024,
                    max_input_image_size=max_input_image_size,
                    offload_model=False,  # Set to True if OOM issues occur
                    **experiment['params']
                )
                
                output_path = os.path.join(concept_output_dir, f"{scenario['name']}.png")
                images[0].save(output_path)
                print(f"Output saved to {output_path}")
                
                total_experiments += 1
                
            except Exception as e:
                print(f"Error running experiment {experiment['name']}: {str(e)}")
                continue
    
    print(f"\nCompleted {total_experiments} identity preservation experiments.")

def run_object_consistency_tests(pipe, output_dir, max_input_image_size):
    """Run object consistency experiments"""
    print("\n=== Running Object Consistency Experiments ===")
    
    experiments = [
        {
            "name": "consistent_object_different_scenes",
            "prompt": "A sks teapot placed on a wooden table in a bright kitchen",
            "input_images": [],
            "params": {
                "guidance_scale": 2.5,
                "seed": 345
            }
        },
        {
            "name": "consistent_object_different_scenes_2",
            "prompt": "A sks teapot placed on a rock in a forest",
            "input_images": [],
            "params": {
                "guidance_scale": 2.5,
                "seed": 345  # Same seed to test consistency
            }
        },
        {
            "name": "consistent_object_interaction",
            "prompt": "A woman is pouring tea from a sks teapot into a cup",
            "input_images": [],
            "params": {
                "guidance_scale": 2.5,
                "seed": 345  # Same seed to test consistency
            }
        }
    ]
    
    _run_experiments(pipe, experiments, os.path.join(output_dir, 'object_consistency'), max_input_image_size)

def run_multi_object_composition_tests(pipe, output_dir, max_input_image_size):
    """Run multi-object composition experiments"""
    print("\n=== Running Multi-Object Composition Experiments ===")
    
    experiments = [
        {
            "name": "two_object_composition",
            "prompt": "A sks backpack next to a sks bear-plushie stuffed animal on a bench",
            "input_images": [],
            "params": {
                "guidance_scale": 2.5,
                "seed": 456
            }
        },
        {
            "name": "three_object_composition",
            "prompt": "A sks Pembroke Welsh Corgi dog wearing sks glasses and sitting next to a sks clock",
            "input_images": [],
            "params": {
                "guidance_scale": 2.5,
                "seed": 567
            }
        },
        {
            "name": "complex_scene_composition",
            "prompt": "A living room with a sks vase on the coffee table, a sks wolf-plushie stuffed animal on the couch, and a sks Russian Blue cat looking at both of them",
            "input_images": [],
            "params": {
                "guidance_scale": 2.5,
                "seed": 678
            }
        }
    ]
    
    _run_experiments(pipe, experiments, os.path.join(output_dir, 'multi_object_composition'), max_input_image_size)

def run_style_transfer_tests(pipe, output_dir, max_input_image_size):
    """Run style transfer experiments"""
    print("\n=== Running Style Transfer Experiments ===")
    
    experiments = [
        {
            "name": "watercolor_style",
            "prompt": "A watercolor painting of a sks backpack-dog in a park",
            "input_images": [],
            "params": {
                "guidance_scale": 2.5,
                "seed": 789
            }
        },
        {
            "name": "comic_style",
            "prompt": "A comic book style drawing of a sks colorful sneaker with superpowers",
            "input_images": [],
            "params": {
                "guidance_scale": 2.5,
                "seed": 890
            }
        },
        {
            "name": "photorealistic_style",
            "prompt": "A photorealistic image of a sks poop-emoji toy in a museum display case",
            "input_images": [],
            "params": {
                "guidance_scale": 2.5,
                "seed": 901
            }
        }
    ]
    
    _run_experiments(pipe, experiments, os.path.join(output_dir, 'style_transfer'), max_input_image_size)

def run_object_manipulation_tests(pipe, output_dir, max_input_image_size):
    """Run object manipulation experiments"""
    print("\n=== Running Object Manipulation Experiments ===")
    
    experiments = [
        {
            "name": "object_scaling",
            "prompt": "A giant sks Corgi puppy dog the size of a car next to a tiny sks bear-plushie stuffed animal the size of a coin",
            "input_images": [],
            "params": {
                "guidance_scale": 2.5,
                "seed": 123
            }
        },
        {
            "name": "object_color_variation",
            "prompt": "A blue sks duck toy next to a red sks duck toy on a table",
            "input_images": [],
            "params": {
                "guidance_scale": 2.5,
                "seed": 234
            }
        },
        {
            "name": "object_transformation",
            "prompt": "A sks candle that is melting into the shape of a sks teapot",
            "input_images": [],
            "params": {
                "guidance_scale": 2.5,
                "seed": 345
            }
        }
    ]
    
    _run_experiments(pipe, experiments, os.path.join(output_dir, 'object_manipulation'), max_input_image_size)

def _run_experiments(pipe, experiments, output_dir, max_input_image_size):
    """Helper function to run experiments"""
    os.makedirs(output_dir, exist_ok=True)
    
    for exp in experiments:
        print(f"\nRunning experiment: {exp['name']}")
        print(f"Prompt: {exp['prompt'][:100]}...")
        
        if exp['input_images']:
            print("Input images:")
            for img_path in exp['input_images']:
                print(f"- {img_path}")
        
        images = pipe(
            prompt=exp['prompt'],
            input_images=exp['input_images'] if 'input_images' in exp and exp['input_images'] else None,
            height=1024,
            width=1024,
            max_input_image_size=max_input_image_size,
            offload_model=False,  # Set to True if OOM issues occur
            **exp['params']
        )
        
        output_path = os.path.join(output_dir, f"{exp['name']}.png")
        images[0].save(output_path)
        print(f"Output saved to {output_path}")

def compute_evaluation_metrics(output_dir):
    """Compute evaluation metrics"""
    print("\n=== Computing Evaluation Metrics ===")
    
    # These would be implemented based on your specific evaluation needs
    # Examples include:
    # - CLIP score for text-image alignment
    # - FID score for image quality
    # - Identity preservation metrics (feature similarity)
    # - Object detection metrics for specific objects
    
    print("Metrics computation placeholder - would implement actual metrics here")

# Example usage

def main():
    args = setup_args()
    from modules.utils import load_para_rank_adapter, add_para_rank_methods
    from modules.pepara import make_para_rank_adapter
    from peft import LoraConfig
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU: {args.gpu}")
    
    # Create output directories
    ensure_output_dir(args.output_dir)
    print(f"Outputs will be saved to: {args.output_dir}")
    
    # Import and load model
    print("Loading OmniGen model...")
    from OmniGen import OmniGenPipeline

    # adapter_path = "/nvme-data/Komal/documents/results/DREAMBOOT/full/qr/checkpoints/0004000/"
    adapter_path = "/nvme-data/Komal/documents/results/DREAMBOOT/full/nmf/checkpoints/0008000/"
    print(f"Loading adapter weights from: {adapter_path}")
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
    print("Model loaded successfully!")

    transformer_lora_config = LoraConfigExtended(
        r=64,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=["qkv_proj", "o_proj"],
        decomposition_method=args.decomposition_method
    )

    pipe.model = load_para_rank_adapter(pipe.model, adapter_path, transformer_lora_config)

    run_comprehensive_evaluation(pipe, args.output_dir)

    print("\n=== All experiments completed successfully! ===")
    print(f"Results are saved in {args.output_dir}")

if __name__ == "__main__":
    main()