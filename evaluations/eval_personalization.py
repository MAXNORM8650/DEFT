import os
import argparse
from PIL import Image
from pathlib import Path
import sys
sys.path.append("/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen")
from functions.image_utils import save_image
from modules.utils import LoraConfigExtended
import os
import torch
import json
def setup_args():
    parser = argparse.ArgumentParser(description='Run OmniGen experiments')
    parser.add_argument('--gpu', type=str, default='1', help='GPU ID to use')
    parser.add_argument('--output_dir', type=str, default='./results/omkar', help='Directory to save outputs')
    parser.add_argument('--max_image_size', type=int, default=1024, help='Maximum input image size')
    parser.add_argument('--decomposition_method', type=str, default='qr', help='Decomposition method to use')
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to adapter weights')
    parser.add_argument('--adapter_type', type=str, required=True, help='Type of adapter to use (e.g.,"DEFT", "para", "lora", "pytorch_lora")')
    return parser.parse_args()

def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)

def run_text2img_experiments(pipe, output_dir):
    """Run text-to-image generation experiments"""
    print("\n=== Running Text-to-Image Experiments ===")
    prompts = generate_personalization_prompts_for_dog_uncond()
    # prompts = [
    #     "a photo of sks dog <img><|image_1|></img> in a lush green field, the dog is playing with a ball, the sun is shining brightly in the background",
    #     "A photo of sks dog <img><|image_1|></img> in beach, the dog is playing with a ball, the sun is setting in the background, waves are crashing on the shore",
    #     "A photo of sks dog <img><|image_1|></img> in a snowy landscape, the dog is playing with a snowball, snowflakes are falling gently from the sky",
    #     "A photo of sks dog <img><|image_1|></img> in a cityscape, the dog is sitting on a sidewalk, skyscrapers are towering in the background, people walking by",
    #     "A photo of sks dog <img><|image_1|></img> in a garden, the dog is sniffing flowers, butterflies are fluttering around, a small pond is visible in the background",
    #     "A photo of sks dog <img><|image_1|></img> in a forest, the dog is running through the trees, sunlight filtering through the leaves, a small stream is flowing nearby",
    # ]
    # input_images = ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"]  # Example input image
    input_images = None
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating image {i+1}/{len(prompts)}")
        print(f"Prompt: {prompt[:100]}...")
        
        images = pipe(
            prompt=prompt,
            input_images=input_images,
            height=1024,
            width=1024,
            guidance_scale=2.5,
            separate_cfg_infer=False,
            seed=0,
        )
        
        output_path = os.path.join(output_dir, 'text2img_uncond', f'output_{i}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        images[0].save(output_path)
        print(f"Image saved to {output_path}")

def run_img2img_experiments(pipe, output_dir, max_input_image_size):
    """Run image-to-image generation experiments"""
    print("\n=== Running Image-to-Image Experiments ===")
    
    # Single image input
    print("\nExperiment: Single image input")
    prompt = "The man in <img><|image_1|></img> waves his hand happily in the to the his girlfriend"
    input_images = ["./imgs/test_cases/Komal.jpeg"]
    
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=1024,
        width=1024,
        guidance_scale=2.5,
        img_guidance_scale=1.8,
        seed=42
    )
    
    # Display input images
    print("Input images:")
    for img_path in input_images:
        print(f"- {img_path}")
    
    # Save output
    output_path = os.path.join(output_dir, 'img2img', 'single_komal1.png')
    images[0].save(output_path)
    print(f"Output saved to {output_path}")
    
    # Multiple images input
    print("\nExperiment: Multiple images input")
    prompt = "The solid man in <img><|image_1|></img> waves his hand happily to his girlfriend in <img><|image_2|></img>."
    input_images = ["./imgs/test_cases/Komal.jpeg", "./imgs/test_cases/Amanda.jpg"]
    
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=1024,
        width=1024,
        guidance_scale=2.5,
        img_guidance_scale=1.8,
        max_input_image_size=max_input_image_size,
        seed=168
    )
    
    # Display input images
    print("Input images:")
    for img_path in input_images:
        print(f"- {img_path}")
    
    # Save output
    output_path = os.path.join(output_dir, 'img2img', 'multiple_images_komal1.png')
    images[0].save(output_path)
    print(f"Output saved to {output_path}")

def run_Personalized_text2img_experiments(pipe, output_dir):
    """
    Run personalized text-to-image generation experiments.
    Note: This function is a placeholder for personalized text-to-image experiments.
    """
    print("\n=== Running Personalized Text-to-Image Experiments ===")
    
    # Example personalized prompt
    prompt = "A photo of sks dog <img><|image_1|></img> in a lush green field, the person is playing with a ball, the sun is shining brightly in the background, in Ghibli style"
    input_images = ["./toy_data/images/dog4.jpeg"]  # Example input image for personalization
    
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=1024,
        width=1024,
        guidance_scale=2.5,
        seed=42
    )
    
    # Save output
    output_path = os.path.join(output_dir, 'personalized_text2img', 'gibili_dog1.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(output_path)
    print(f"Output saved to {output_path}")
def run_KPersonalized_text2img_experiments(pipe, output_dir):
    """
    Run personalized text-to-image generation experiments.
    Note: This function is a placeholder for personalized text-to-image experiments.
    """
    print("\n=== Running Personalized Text-to-Image Experiments ===")
    
    # Example personalized prompt
    prompt = "A photo of sks Komal <img><|image_1|></img>, whimsical Ghibli Studio animation style, featuring vibrant colors, dreamy lighting, and detailed character expression"
    input_images = ["./toy_data/Komal_p/File 5.jpeg"]  # Example input image for personalization
    
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=1024,
        width=1024,
        guidance_scale=2.5,
        seed=42
    )
    
    # Save output
    output_path = os.path.join(output_dir, 'personalized_text2img', 'Plus_gibli_Komal5.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(output_path)
    print(f"Output saved to {output_path}")
def run_Schene_Personalized_text2img_experiments(pipe, output_dir):
    """
    Run personalized text-to-image generation experiments.
    Note: This function is a placeholder for personalized text-to-image experiments.
    """
    print("\n=== Running Personalized Text-to-Image Experiments ===")
    prompt = "A modern, minimalistic office workspace with a white round table at the center. On the table, a transparent plastic bottle filled with orange sits upright, with a black cap. Next to it, a beige cap rests with its brim facing downward. A jar of protien with a orange lid sits adjacent to the cap, slightly worn as if frequently opened. Beside the jar, a black wallet rests on the table, and next to it is a clear plastic container holding crumpled receipts and notes. Slightly to the right, a black cylindrical drink container stands upright with visible branding, possibly containing coffee or another beverage. A black computer mouse lies flat on the table near the container. A purple-handled comb lies horizontally across the table, its teeth directed left. In the background, a blue office chair is fully visible, enhancing the casual yet professional atmosphere. The lighting is soft, casting a warm glow over the table and objects, with a faint computer screen slightly out of view in the background, indicating the workspace’s function. The scene conveys a balance between productivity and relaxation, with personal and work items scattered across the table in a well-organized but casual manner."
        # Example personalized prompt
    prompt = "A photo of sks a photo of sks leisure zone <img><|image_1|></img>, whimsical Ghibli Studio animation style, featuring vibrant colors, dreamy lighting, and detailed character expression"
    input_images = ["/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/Scenes/easy/leisure_zone_001/rgb_000.jpg"]  # Example input image for personalization same as other path 
    
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=1024,
        width=1024,
        guidance_scale=2.5,
        seed=42
    )
    
    # Save output
    output_path = os.path.join(output_dir, 'personalized_text2img/scnene', 'gibli.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(output_path)
    print(f"Output saved to {output_path}")
def run_identity_preservation_experiments(pipe, output_dir, max_input_image_size):
    """Run identity preservation experiments"""
    print("\n=== Running Identity Preservation Experiments ===")
    # generate identity preservation experiments
    prompt = "A man in a black shirt is reading a book and a sks dog is siting on top of his head"
    experiments = [
        {
            "name": "single_object",
            "prompt": "A man in a black shirt is reading a book and a sks Corgi puppy dog <img><|image_1|></img> is siting in top of his head. The man is the right man in <img><|image_2|></img>.",
            "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog5.jpeg", "./imgs/test_cases/two_man.jpg"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.6,
                "seed": 0
            }
        },
        {
            "name": "multiple_objects",
            "prompt": "A man is sitting in the library reading a book, while a woman wearing white shirt next to him putting headphone on a sks Corgi puppy dog. The man who is reading is the one wearing red sweater in <img><|image_1|></img>. The woman putting headphones is the right woman wearing suit in <img><|image_2|></img>. The sks Corgi puppy dog is the one in <img><|image_3|></img>.",
            "input_images": ["./imgs/test_cases/turing.png", "./imgs/test_cases/lecun.png", "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog5.jpeg"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.8,
                "seed": 2
            }
        },
        {
            "name": "Komal_bookshelf_scene",
            "prompt": "A sks Corgi puppy dog and a short-haired woman with a wrinkled face are standing in front of a bookshelf in a library. The sks Corgi puppy dog is <img><|image_1|></img>, and the woman is oldest woman in <img><|image_2|></img>",
            "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog5.jpeg", "./imgs/test_cases/2.jpg"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.6,
                "seed": 60
            }
        },
        {
            "name": "classroom_scene",
            "prompt": "A man and a woman are sitting at a classroom desk teaching a sks Corgi puppy dog. The man is the man with yellow hair in <img><|image_1|></img>. The woman is the woman on the left of <img><|image_2|></img>. The sks Corgi puppy dog is the one in <img><|image_3|></img>.",
            "input_images": ["./imgs/test_cases/3.jpg", "./imgs/test_cases/4.jpg", "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog5.jpeg"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.6,
                "seed": 66
            }
        },
        {
            "name": "fashion_scene",
            "prompt": "A woman is walking down the street walking a sks Corgi puppy dog, wearing a white long-sleeve blouse with lace details on the sleeves, paired with a blue pleated skirt. The woman is <img><|image_1|></img>. The long-sleeve blouse and a pleated skirt are <img><|image_2|></img>. The sks Corgi puppy dog is the one in <img><|image_3|></img>.",
            "input_images": ["./imgs/demo_cases/emma.jpeg", "./imgs/demo_cases/dress.jpg", "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog5.jpeg"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.6,
                "seed": 666
            }
        }
    ]
    
    for exp in experiments:
        print(f"\nRunning experiment: {exp['name']}")
        print(f"Prompt: {exp['prompt'][:100]}...")
        print("Input images:")
        for img_path in exp['input_images']:
            print(f"- {img_path}")
            
        images = pipe(
            prompt=exp['prompt'],
            input_images=exp['input_images'],
            height=1024,
            width=1024,
            max_input_image_size=max_input_image_size,
            offload_model=False,  # Set to True if OOM issues occur
            **exp['params']
        )
        
        output_path = os.path.join(output_dir, 'identity_preservation', f"{exp['name']}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        images[0].save(output_path)
        print(f"Output saved to {output_path}")

def run_image_conditioning_experiments(pipe, output_dir):
    """Run image conditioning experiments"""
    print("\n=== Running Image Conditioning Experiments ===")
    
    experiments = [
        {
            "name": "skeleton_detection",
            "prompt": "Detect the skeleton of human in this image: <img><|image_1|></img>.",
            "input_images": ["./imgs/test_cases/control.jpg"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.6,
                "separate_cfg_infer": False,
                "seed": 0
            }
        },
        {
            "name": "condition_based_generation",
            "prompt": "Generate a new photo using the following picture and text as conditions: <img><|image_1|><img>\n An elderly man wearing gold-framed glasses stands dignified in front of an elegant villa. His gray hair is neatly combed, and his hands rest in the pockets of his dark trousers. He is dressed warmly in a fitted coat over a sweater. The classic villa behind him features ivy-covered walls and large bay windows.",
            "input_images": ["./imgs/test_cases/pose.png"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.6,
                "seed": 0
            }
        },
        {
            "name": "human_pose_following",
            "prompt": "Following the human pose of this image <img><|image_1|></img>, generate a new photo: An elderly man wearing a gold-framed glasses stands dignified in front of an elegant villa. His gray hair is neatly combed, and his hands rest in the pockets of his dark trousers. He is dressed warmly in a fitted coat over a sweater. The classic villa behind him features ivy-covered walls and large bay windows.",
            "input_images": ["./imgs/test_cases/control.jpg"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.6,
                "seed": 0
            }
        },
        {
            "name": "reasoning_remove_watch",
            "prompt": "<img><|image_1|><\/img> What item can be used to see the current time? Please remove it.",
            "input_images": ["./imgs/test_cases/watch.jpg"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.6,
                "seed": 0
            }
        }
    ]
    
    for exp in experiments:
        print(f"\nRunning experiment: {exp['name']}")
        print(f"Prompt: {exp['prompt'][:100]}...")
        print("Input images:")
        for img_path in exp['input_images']:
            print(f"- {img_path}")
            
        images = pipe(
            prompt=exp['prompt'],
            input_images=exp['input_images'],
            height=1024,
            width=1024,
            **exp['params']
        )
        
        output_path = os.path.join(output_dir, 'img_conditioning', f"{exp['name']}.png")
        images[0].save(output_path)
        print(f"Output saved to {output_path}")
def run_reasoning_experiments(pipe, output_dir):
    """Run reasoning experiments"""
    print("\n=== Running Reasoning Experiments ===")
    
    # Example reasoning experiment
    prompt = "<img><|image_1|><\/img> What item can be used to see the current time? Please remove it."
    input_images = ["./imgs/test_cases/watch.jpg"]
    
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=1024,
        width=1024,
        guidance_scale=2.5,
        img_guidance_scale=1.6,
        seed=0
    )
    
    # Display input images
    print("Input images:")
    for img_path in input_images:
        print(f"- {img_path}")
    
    # Save output
    output_path = os.path.join(output_dir, 'reasoning', 'remove_watch.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(output_path)
    print(f"Output saved to {output_path}")

def run_objcet_mask_generation_experiments(pipe, output_dir):
    """Run object mask generation experiments"""
    print("\n=== Running Object Mask Generation Experiments ===")
    
    # Example object mask generation experiment
    prompt = "A segmentation mask of a photo of sks leisure zone <img><|image_1|></img>."
    input_images = ["/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/Scenes/easy/leisure_zone_001/rgb_000.jpg"]  # Example input image for mask generation
    
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=1024,
        width=1024,
        guidance_scale=2.5,
        img_guidance_scale=1.6,
        seed=0
    )
    
    # Display input images
    print("Input images:")
    for img_path in input_images:
        print(f"- {img_path}")
    
    # Save output
    output_path = os.path.join(output_dir, 'object_mask_generation_adaptor', 'easy-leisure_zone_001.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(output_path)
    print(f"Output saved to {output_path}")
def run_depth_map_generation_experiments(pipe, output_dir):
    """Run depth map generation experiments"""
    print("\n=== Running Depth Map Generation Experiments ===")
    
    # Example depth map generation experiment
    prompt = "A depth map of a photo of sks leisure zone <img><|image_1|></img>."
    input_images = ["/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/Scenes/easy/leisure_zone_001/rgb_000.jpg"]  # Example input image for depth map generation
    
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=1024,
        width=1024,
        guidance_scale=2.5,
        img_guidance_scale=1.6,
        seed=0
    )
    
    # Display input images
    print("Input images:")
    for img_path in input_images:
        print(f"- {img_path}")
    
    # Save output
    output_path = os.path.join(output_dir, 'depth_map_generation_adaptor', 'easy-leisure_zone_001_depth.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(output_path)
    print(f"Output saved to {output_path}")
def run_lays_chip_experiments(pipe, output_dir):
    """Run lays chip experiments"""
    print("\n=== Running Lays Chip Experiments ===")
    
    # Example lays chip experiment
    prompt = "A photo of sks lays chip <img><|image_1|></img>."
    input_images = ["/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/Scenes/easy/leisure_zone_001/rgb_000.jpg"]  # Example input image for lays chip
    
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=1024,
        width=1024,
        guidance_scale=2.5,
        img_guidance_scale=1.6,
        seed=0
    )
    
    # Display input images
    print("Input images:")
    for img_path in input_images:
        print(f"- {img_path}")
    
    # Save output
    output_path = os.path.join(output_dir, 'lays_chip_adaptor', 'easy-leisure_zone_001_lays_chip.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(output_path)
    print(f"Output saved to {output_path}")

def run_lays_experiments(pipe, output_dir):
    # Example object detection experiment
    prompt = "sks lays chip tube auburn <img><|image_1|></img> in beach."
    input_images = ["/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/Objects/097_lays_chip_tube_auburn/images/003.jpg"]  # Example input image for object detection
    
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=1024,
        width=1024,
        guidance_scale=2.5,
        img_guidance_scale=1.6,
        seed=0
    )
    
    # Display input images
    print("Input images:")
    for img_path in input_images:
        print(f"- {img_path}")
    
    # Save output
    output_path = os.path.join(output_dir, 'single_object', 'beach_097_lays_chip_img3.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(output_path)
    print(f"Output saved to {output_path}")

def running_multiple_prompt_experiments(pipe, output_dir):
    """Run multiple prompt experiments"""
    print("\n=== Running Multiple Prompt Experiments ===")
    
    # Example multiple prompt experiment
    prompt = "A photo of sks lays chip tube auburn <img><|image_1|></img> in top of sks lays chip tube auburn <img><|image_2|></img> in beach."
    input_images = ["/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/Objects/097_lays_chip_tube_auburn/images/003.jpg", "/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/Objects/097_lays_chip_tube_auburn/images/004.jpg"]  # Example input image for multiple prompts
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=1024,
        width=1024,
        guidance_scale=2.5,
        img_guidance_scale=1.6,
        seed=0
    )
    # Display input images
    print("Input images:")
    for img_path in input_images:
        print(f"- {img_path}")
    # Save output
    output_path = os.path.join(output_dir, 'multiple_prompt', '097_lays_chip_img3_on_img4.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(output_path)
    print(f"Output saved to {output_path}") 



import os
import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import json

def evaluate_image_quality(original_img_path, generated_img_path):
    """Evaluates the generated image based on SSIM and PSNR metrics"""
    original_img = np.array(Image.open(original_img_path))
    generated_img = np.array(Image.open(generated_img_path))
    
    ssim_value = ssim(original_img, generated_img, multichannel=True)
    psnr_value = psnr(original_img, generated_img)
    
    return ssim_value, psnr_value

def generate_personalization_prompts():
    """Generates various personalization prompts to test the model under different conditions."""
    prompts = [
        "a realistic photo of a sks lays chip tube auburn <img><|image_1|></img> on a beach during sunset",
        "a close-up photo of a sks lays chip tube auburn <img><|image_1|></img> in a kitchen with warm lighting",
        "a photo of a sks lays chip tube auburn <img><|image_1|></img> on a snowy mountain top",
        "a photo of a sks lays chip tube auburn <img><|image_1|></img> floating in a pool surrounded by palm trees",
        "an artistic rendering of a sks lays chip tube auburn <img><|image_1|></img> in a futuristic city at night",
        "a sks lays chip tube auburn <img><|image_1|></img> in a crowded street market with vibrant colors and lights",
        "a sks lays chip tube auburn <img><|image_1|></img> on a deserted island with a clear blue sky and calm sea",
        "a photo of a sks lays chip tube auburn <img><|image_1|></img> surrounded by a leaves in a forest",
        "a stylized photo of a sks lays chip tube auburn <img><|image_1|></img> on a rainy day with raindrops visible",
        "a macro shot of a sks lays chip tube auburn <img><|image_1|></img> on a dark background with dramatic shadows"
    ]
    return prompts

def generate_personalization_prompts_for_dog():
    """Generates various personalization prompts to test the model under different conditions."""
    prompts = [
        "a realistic photo of a sks dog <img><|image_1|></img> on a beach during sunset",
        "a close-up photo of a sks dog <img><|image_1|></img> in a kitchen with warm lighting",
        "a photo of a sks dog <img><|image_1|></img> on a snowy mountain top",
        "a photo of a sks ladog <img><|image_1|></img> floating in a pool surrounded by palm trees",
        "an artistic rendering of a sks dog <img><|image_1|></img> in a futuristic city at night",
        "a sks dog <img><|image_1|></img> in a crowded street market with vibrant colors and lights",
        "a sks dog <img><|image_1|></img> on a deserted island with a clear blue sky and calm sea",
        "a photo of a sks dog <img><|image_1|></img> surrounded by autumn leaves in a forest",
        "a stylized photo of a sks dog <img><|image_1|></img> on a rainy day with raindrops visible",
        "a macro shot of a sks dog <img><|image_1|></img> on a dark background with dramatic shadows"
    ]
    return prompts
def generate_personalization_prompts_for_dog_uncond():
    """Generates various personalization prompts to test the model under different conditions."""
    prompts = [
        "a realistic photo of a sks dog on a beach during sunset",
        "a close-up photo of a sks dog in a kitchen with warm lighting",
        "a photo of a sks dog on a snowy mountain top",
        "a photo of a sks ladog floating in a pool surrounded by palm trees",
        "an artistic rendering of a sks dog in a futuristic city at night",
        "a sks dog in a crowded street market with vibrant colors and lights",
        "a sks dog on a deserted island with a clear blue sky and calm sea",
        "a photo of a sks dog surrounded by autumn leaves in a forest",
        "a stylized photo of a sks dog on a rainy day with raindrops visible",
        "a macro shot of a sks dog on a dark background with dramatic shadows"
    ]
    return prompts
def generate_personalization_prompts_for_SEEN():
    """Generates various personalization prompts to test the model under different conditions."""
    prompts = [
        "a realistic photo of a sks Church Roc <img><|image_1|></img> on a beach during sunset",
        "a close-up photo of a sks Church Roc <img><|image_1|></img> in a kitchen with warm lighting",
        "a photo of asks Church Roc <img><|image_1|></img> on a snowy mountain top",
        "a photo of a sks Church Roc <img><|image_1|></img> floating in a pool surrounded by palm trees",
        "an artistic rendering of a sks Church Roc <img><|image_1|></img> in a futuristic city at night",
        "a photo of sks Church Roc <img><|image_1|></img> in a crowded street market with vibrant colors and lights",
        "a photo of sks Church Roc <img><|image_1|></img> on a deserted island with a clear blue sky and calm sea",
        "a photo of sks Church Roc <img><|image_1|></img> surrounded by autumn leaves in a forest",
        "a stylized photo of a sks Church Roc <img><|image_1|></img> on a rainy day with raindrops visible",
        "a macro shot of a sks Church Roc <img><|image_1|></img> on a dark background with dramatic shadows"
    ]
    return prompts
def generate_personalization_prompts_for_SFM_SEEN():
    """Generates various personalization prompts to test the model under different conditions."""
    prompts = [
        "a realistic photo of a sks Church Rock on a beach during sunset <img><|image_1|></img>, now transform into different view with camera info: {'model': 'Camera', 'width': 3840, 'height': 2160, 'params': [3064.1587249981353, 1920.0, 1080.0, -0.012093892585680123]} cam_from_world: Rigid3d(rotation_xyzw=[-0.0689793, -0.783644, -0.323276, 0.525963], translation=[0.0595202, -1.68451, 4.33625]), projection center: [-4.13261501 -0.59266592  2.05290163], viewing direction: [ 0.86893389  0.43410506 -0.23771135]",
        "a close-up a photo of a sks Church Rock in a kitchen with warm lighting on a beach during sunset <img><|image_1|></img>, now transform into different view with camera info: {'model': 'Camera', 'width': 3840, 'height': 2160, 'params': [3064.1587249981353, 1920.0, 1080.0, -0.012093892585680123]} cam_from_world: Rigid3d(rotation_xyzw=[-0.0689793, -0.783644, -0.323276, 0.525963], translation=[0.0595202, -1.68451, 4.33625]), projection center: [-4.13261501 -0.59266592  2.05290163], viewing direction: [ 0.86893389  0.43410506 -0.23771135]",
        "a photo of a photo of a sks Church Rock on a beach during sunset <img><|image_1|></img> on a snowy mountain top, now transform into different view with camera info: {'model': 'Camera', 'width': 3840, 'height': 2160, 'params': [3064.1587249981353, 1920.0, 1080.0, -0.012093892585680123]} cam_from_world: Rigid3d(rotation_xyzw=[-0.0689793, -0.783644, -0.323276, 0.525963], translation=[0.0595202, -1.68451, 4.33625]), projection center: [-4.13261501 -0.59266592  2.05290163], viewing direction: [ 0.86893389  0.43410506 -0.23771135]",
        "a photo of a photo of a sks Church Rock on a beach during sunset <img><|image_1|></img> floating in a pool surrounded by palm trees, now transform into different view with camera info: {'model': 'Camera', 'width': 3840, 'height': 2160, 'params': [3064.1587249981353, 1920.0, 1080.0, -0.012093892585680123]} cam_from_world: Rigid3d(rotation_xyzw=[-0.0689793, -0.783644, -0.323276, 0.525963], translation=[0.0595202, -1.68451, 4.33625]), projection center: [-4.13261501 -0.59266592  2.05290163], viewing direction: [ 0.86893389  0.43410506 -0.23771135]",
        "an artistic rendering of a photo of sks Church Rock on a beach during sunset <img><|image_1|></img> in a futuristic city at night, now transform into different view with camera info: {'model': 'Camera', 'width': 3840, 'height': 2160, 'params': [3064.1587249981353, 1920.0, 1080.0, -0.012093892585680123]} cam_from_world: Rigid3d(rotation_xyzw=[-0.0689793, -0.783644, -0.323276, 0.525963], translation=[0.0595202, -1.68451, 4.33625]), projection center: [-4.13261501 -0.59266592  2.05290163], viewing direction: [ 0.86893389  0.43410506 -0.23771135]",
        "a photo of a sks Church Rock in a crowded street market with vibrant colors and lights on a beach during sunset <img><|image_1|></img>, now transform into different view with camera info: {'model': 'Camera', 'width': 3840, 'height': 2160, 'params': [3064.1587249981353, 1920.0, 1080.0, -0.012093892585680123]} cam_from_world: Rigid3d(rotation_xyzw=[-0.0689793, -0.783644, -0.323276, 0.525963], translation=[0.0595202, -1.68451, 4.33625]), projection center: [-4.13261501 -0.59266592  2.05290163], viewing direction: [ 0.86893389  0.43410506 -0.23771135]",
        "a photo of a sks Church Rock on a beach during sunset on a deserted island with a clear blue sky and calm sea  on a beach during sunset <img><|image_1|></img> on a deserted island with a clear blue sky and calm sea, now transform into different view with camera info: {'model': 'Camera', 'width': 3840, 'height': 2160, 'params': [3064.1587249981353, 1920.0, 1080.0, -0.012093892585680123]} cam_from_world: Rigid3d(rotation_xyzw=[-0.0689793, -0.783644, -0.323276, 0.525963], translation=[0.0595202, -1.68451, 4.33625]), projection center: [-4.13261501 -0.59266592  2.05290163], viewing direction: [ 0.86893389  0.43410506 -0.23771135]",
        "a photo of a sks Church Rock on a beach during sunset <img><|image_1|></img> surrounded by autumn leaves in a forest, now transform into different view with camera info: {'model': 'Camera', 'width': 3840, 'height': 2160, 'params': [3064.1587249981353, 1920.0, 1080.0, -0.012093892585680123]} cam_from_world: Rigid3d(rotation_xyzw=[-0.0689793, -0.783644, -0.323276, 0.525963], translation=[0.0595202, -1.68451, 4.33625]), projection center: [-4.13261501 -0.59266592  2.05290163], viewing direction: [ 0.86893389  0.43410506 -0.23771135]",
        "a stylized photo of a photo of sks Church Rock on a beach during sunset <img><|image_1|></img> on a rainy day with raindrops visible, now transform into different view with camera info: {'model': 'Camera', 'width': 3840, 'height': 2160, 'params': [3064.1587249981353, 1920.0, 1080.0, -0.012093892585680123]} cam_from_world: Rigid3d(rotation_xyzw=[-0.0689793, -0.783644, -0.323276, 0.525963], translation=[0.0595202, -1.68451, 4.33625]), projection center: [-4.13261501 -0.59266592  2.05290163], viewing direction: [ 0.86893389  0.43410506 -0.23771135]",
        "a macro shot of a photo of a sks Church Rock on a beach during sunset <img><|image_1|></img> on a dark background with dramatic shadows, now transform into different view with camera info: {'model': 'Camera', 'width': 3840, 'height': 2160, 'params': [3064.1587249981353, 1920.0, 1080.0, -0.012093892585680123]} cam_from_world: Rigid3d(rotation_xyzw=[-0.0689793, -0.783644, -0.323276, 0.525963], translation=[0.0595202, -1.68451, 4.33625]), projection center: [-4.13261501 -0.59266592  2.05290163], viewing direction: [ 0.86893389  0.43410506 -0.23771135]"
    ]
    return prompts
# def run_personalization_experiments(pipe, output_dir):
#     """Runs the personalization experiments with multiple prompts and evaluates the results"""
#     prompts = generate_personalization_prompts_for_dog()
    
#     # Loop over each prompt
#     eval_results = {}
#     for idx, prompt in enumerate(prompts):
#         input_images = ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"]
        
#         # Generate output image using the pipe
#         images = pipe(
#             prompt=prompt,
#             input_images=input_images,
#             height=1024,
#             width=1024,
#             guidance_scale=2.5,
#             img_guidance_scale=1.6,
#             seed=0
#         )
        
#         # Save the generated image
#         output_path = os.path.join(output_dir, f'personalized_image_{idx}.png')
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         images[0].save(output_path)
#         print(f"Output saved to {output_path}")
        
        # Evaluate the generated image
        # ssim_value, psnr_value = evaluate_image_quality(reference_image_path, output_path)
        
        # # Store evaluation results
        # eval_results[f"image_{idx}"] = {
        #     "prompt": prompt,
        #     "ssim": ssim_value,
        #     "psnr": psnr_value
        # }
    
    # # Save all evaluation results
    # eval_output_path = os.path.join(output_dir, 'personalization_evaluation_results.json')
    # with open(eval_output_path, 'w') as eval_file:
    #     json.dump(eval_results, eval_file, indent=4)
    
    # print(f"Evaluation results saved to {eval_output_path}")
import os
import torch
import clip
import numpy as np
from PIL import Image
import json
from datetime import datetime
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.stats import entropy
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import cv2
import lpips

def load_evaluation_models():
    """Load models needed for evaluation"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    # Load CLIP from transformers (alternative)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model_hf = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    # Load LPIPS model for perceptual similarity
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    return {
        'clip_model': clip_model,
        'clip_preprocess': clip_preprocess,
        'clip_processor': clip_processor,
        'clip_model_hf': clip_model_hf,
        'lpips_model': lpips_model,
        'device': device
    }

def compute_clip_i_score(input_image_path, generated_image_path, models):
    """Compute CLIP-I score (image-to-image similarity)"""
    device = models['device']
    clip_model = models['clip_model']
    preprocess = models['clip_preprocess']
    
    # Load and preprocess images
    input_image = preprocess(Image.open(input_image_path)).unsqueeze(0).to(device)
    generated_image = preprocess(Image.open(generated_image_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        input_features = clip_model.encode_image(input_image)
        generated_features = clip_model.encode_image(generated_image)
        
        # Normalize features
        input_features = F.normalize(input_features, p=2, dim=1)
        generated_features = F.normalize(generated_features, p=2, dim=1)
        
        # Compute cosine similarity
        clip_i_score = torch.cosine_similarity(input_features, generated_features).item()
    
    return clip_i_score

def compute_clip_t_score(prompt, generated_image_path, models):
    """Compute CLIP-T score (text-to-image similarity)"""
    device = models['device']
    clip_model = models['clip_model']
    preprocess = models['clip_preprocess']
    
    # Load and preprocess image
    generated_image = preprocess(Image.open(generated_image_path)).unsqueeze(0).to(device)
    
    # Tokenize text
    text_tokens = clip.tokenize([prompt]).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(generated_image)
        text_features = clip_model.encode_text(text_tokens)
        
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # Compute cosine similarity
        clip_t_score = torch.cosine_similarity(image_features, text_features).item()
    
    return clip_t_score

def compute_dinv1_score(input_image_path, generated_image_path, models):
    """Compute DINV-1 score (perceptual similarity using LPIPS)"""
    device = models['device']
    lpips_model = models['lpips_model']
    
    # Load images
    input_img = Image.open(input_image_path).convert('RGB')
    generated_img = Image.open(generated_image_path).convert('RGB')
    
    # Resize to same dimensions if needed
    size = (256, 256)
    input_img = input_img.resize(size)
    generated_img = generated_img.resize(size)
    
    # Convert to tensors and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(input_img).unsqueeze(0).to(device)
    generated_tensor = transform(generated_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # LPIPS distance (lower is better, so we return 1 - distance for similarity)
        lpips_distance = lpips_model(input_tensor, generated_tensor).item()
        dinv1_score = 1.0 - lpips_distance  # Convert to similarity score
    
    return dinv1_score

def compute_image_quality_metrics(image_path):
    """Compute various image quality metrics"""
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    metrics = {}
    
    # Basic image properties
    metrics['resolution'] = f"{img.size[0]}x{img.size[1]}"
    metrics['file_size_kb'] = os.path.getsize(image_path) / 1024
    metrics['channels'] = len(img.getbands())
    metrics['mode'] = img.mode
    
    # Image statistics
    if len(img_array.shape) == 3:  # Color image
        metrics['mean_brightness'] = np.mean(img_array)
        metrics['std_brightness'] = np.std(img_array)
        metrics['contrast'] = np.std(img_array) / np.mean(img_array) if np.mean(img_array) > 0 else 0
        
        # Color distribution
        metrics['mean_rgb'] = [float(np.mean(img_array[:,:,i])) for i in range(3)]
        metrics['std_rgb'] = [float(np.std(img_array[:,:,i])) for i in range(3)]
    else:
        metrics['mean_brightness'] = np.mean(img_array)
        metrics['std_brightness'] = np.std(img_array)
        metrics['contrast'] = np.std(img_array) / np.mean(img_array) if np.mean(img_array) > 0 else 0
    
    # Sharpness measure (using Laplacian variance)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    metrics['sharpness'] = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    metrics['edge_density'] = np.sum(edges > 0) / edges.size
    
    return metrics

def compute_aesthetic_score(image_path, models):
    """Compute a simple aesthetic score using CLIP"""
    device = models['device']
    clip_model = models['clip_model']
    preprocess = models['clip_preprocess']
    
    # Aesthetic prompts
    positive_prompts = [
        "a beautiful high quality image",
        "aesthetically pleasing artwork",
        "professional photography",
        "visually appealing composition"
    ]
    
    negative_prompts = [
        "low quality blurry image",
        "ugly distorted picture", 
        "poor composition",
        "amateur photography"
    ]
    
    # Load and preprocess image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image_features = F.normalize(image_features, p=2, dim=1)
        
        # Compute similarity with positive prompts
        pos_scores = []
        for prompt in positive_prompts:
            text_tokens = clip.tokenize([prompt]).to(device)
            text_features = clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=1)
            score = torch.cosine_similarity(image_features, text_features).item()
            pos_scores.append(score)
        
        # Compute similarity with negative prompts
        neg_scores = []
        for prompt in negative_prompts:
            text_tokens = clip.tokenize([prompt]).to(device)
            text_features = clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=1)
            score = torch.cosine_similarity(image_features, text_features).item()
            neg_scores.append(score)
    
    # Aesthetic score as positive minus negative
    aesthetic_score = np.mean(pos_scores) - np.mean(neg_scores)
    
    return {
        'aesthetic_score': aesthetic_score,
        'positive_similarity': np.mean(pos_scores),
        'negative_similarity': np.mean(neg_scores)
    }

def run_personalization_experiments_omkar(pipe, output_dir):
    """Runs the personalization experiments with multiple prompts and evaluates the results"""
    
    # Load evaluation models
    print("Loading evaluation models...")
    # eval_models = load_evaluation_models()

    prompts = [
        "Transform the man depicted in <img><|image_1|></img>, <img><|image_2|></img>, and <img><|image_3|></img> into the likeness of the man shown in <img><|image_4|></img>, who is Mahatma Gandhi. The output should resemble Gandhi during the British colonial period, specifically around the time of the Dandi March in 1947. Ensure the man is bald with a fringe of white hair, wearing iconic round wire-rimmed glasses, and dressed in simple white Indian attire — a cotton dhoti and shawl. He should appear thin, barefoot or in basic sandals, with a gentle and calm facial expression reflecting wisdom and humility. Maintain Gandhi’s signature posture and demeanor, and situate him in a natural, realistic pose aligned with the historical setting. Prioritize authenticity, facial transformation, clothing style, and cultural accuracy."
    ]
    # Loop over each prompt
    
    for idx, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {idx+1}/{len(prompts)}: {prompt[:50]}...")
        # dir  = "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/omkargandhi"

        input_images = ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/omkargandhi/WhatsApp Image 2025-08-05 at 16.55.47 (1).jpeg", "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/omkargandhi/WhatsApp Image 2025-08-05 at 16.55.47.jpeg", "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/omkargandhi/WhatsApp Image 2025-08-05 at 16.55.48.jpeg", "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/omkargandhi/gandhi.jpg"]
        
        # Generate output image using the pipe
        images = pipe(
            prompt=prompt,
            input_images=input_images,
            height=1024,
            width=1024,
            guidance_scale=2.5,
            img_guidance_scale=1.6,
            seed=0
        )
        
        # Save the generated image
        output_path = os.path.join(output_dir, f'personalized_gandhiji_plus_{idx}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        images[0].save(output_path)
        print(f"Output saved to {output_path}")
    

def run_personalization_experiments(pipe, output_dir):
    """Runs the personalization experiments with multiple prompts and evaluates the results"""
    
    # Load evaluation models
    print("Loading evaluation models...")
    eval_models = load_evaluation_models()
    
    prompts = generate_personalization_prompts_for_dog()

    # Loop over each prompt
    eval_results = {}
    all_metrics = []
    
    for idx, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {idx+1}/{len(prompts)}: {prompt[:50]}...")
        
        input_images = ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"]
        
        # Generate output image using the pipe
        images = pipe(
            prompt=prompt,
            input_images=input_images,
            height=1024,
            width=1024,
            guidance_scale=2.5,
            img_guidance_scale=1.6,
            seed=0
        )
        
        # Save the generated image
        output_path = os.path.join(output_dir, f'personalized_image_{idx}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        images[0].save(output_path)
        print(f"Output saved to {output_path}")
        
        # Evaluate the generated image
        print("Computing evaluation metrics...")
        
        try:
            # CLIP-I Score (Image-to-Image similarity)
            clip_i = compute_clip_i_score(input_images[0], output_path, eval_models)
            
            # CLIP-T Score (Text-to-Image similarity)
            clip_t = compute_clip_t_score(prompt, output_path, eval_models)
            
            # DINV-1 Score (Perceptual similarity)
            dinv1 = compute_dinv1_score(input_images[0], output_path, eval_models)
            
            # Image quality metrics
            quality_metrics = compute_image_quality_metrics(output_path)
            
            # Aesthetic score
            aesthetic_metrics = compute_aesthetic_score(output_path, eval_models)
            
            # Compile results for this prompt
            prompt_results = {
                'prompt_idx': idx,
                'prompt': prompt,
                'output_path': output_path,
                'clip_i_score': clip_i,
                'clip_t_score': clip_t,
                'dinv1_score': dinv1,
                'quality_metrics': quality_metrics,
                'aesthetic_metrics': aesthetic_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            eval_results[f'prompt_{idx}'] = prompt_results
            all_metrics.append(prompt_results)
            
            print(f"  CLIP-I: {clip_i:.4f}")
            print(f"  CLIP-T: {clip_t:.4f}")
            print(f"  DINV-1: {dinv1:.4f}")
            print(f"  Aesthetic: {aesthetic_metrics['aesthetic_score']:.4f}")
            print(f"  Sharpness: {quality_metrics['sharpness']:.2f}")
            
        except Exception as e:
            print(f"Error computing metrics for prompt {idx}: {e}")
            eval_results[f'prompt_{idx}'] = {
                'prompt_idx': idx,
                'prompt': prompt,
                'output_path': output_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Compute aggregate statistics
    if all_metrics:
        print("\n" + "="*50)
        print("AGGREGATE RESULTS")
        print("="*50)
        
        clip_i_scores = [m['clip_i_score'] for m in all_metrics if 'clip_i_score' in m]
        clip_t_scores = [m['clip_t_score'] for m in all_metrics if 'clip_t_score' in m]
        dinv1_scores = [m['dinv1_score'] for m in all_metrics if 'dinv1_score' in m]
        aesthetic_scores = [m['aesthetic_metrics']['aesthetic_score'] for m in all_metrics if 'aesthetic_metrics' in m]
        sharpness_scores = [m['quality_metrics']['sharpness'] for m in all_metrics if 'quality_metrics' in m]
        
        aggregate_stats = {
            'clip_i': {
                'mean': np.mean(clip_i_scores),
                'std': np.std(clip_i_scores),
                'min': np.min(clip_i_scores),
                'max': np.max(clip_i_scores)
            },
            'clip_t': {
                'mean': np.mean(clip_t_scores),
                'std': np.std(clip_t_scores),
                'min': np.min(clip_t_scores),
                'max': np.max(clip_t_scores)
            },
            'dinv1': {
                'mean': np.mean(dinv1_scores),
                'std': np.std(dinv1_scores),
                'min': np.min(dinv1_scores),
                'max': np.max(dinv1_scores)
            },
            'aesthetic': {
                'mean': np.mean(aesthetic_scores),
                'std': np.std(aesthetic_scores),
                'min': np.min(aesthetic_scores),
                'max': np.max(aesthetic_scores)
            },
            'sharpness': {
                'mean': np.mean(sharpness_scores),
                'std': np.std(sharpness_scores),
                'min': np.min(sharpness_scores),
                'max': np.max(sharpness_scores)
            }
        }
        
        eval_results['aggregate_statistics'] = aggregate_stats
        
        # Print summary
        print(f"CLIP-I Score:    {aggregate_stats['clip_i']['mean']:.4f} ± {aggregate_stats['clip_i']['std']:.4f}")
        print(f"CLIP-T Score:    {aggregate_stats['clip_t']['mean']:.4f} ± {aggregate_stats['clip_t']['std']:.4f}")
        print(f"DINV-1 Score:    {aggregate_stats['dinv1']['mean']:.4f} ± {aggregate_stats['dinv1']['std']:.4f}")
        print(f"Aesthetic Score: {aggregate_stats['aesthetic']['mean']:.4f} ± {aggregate_stats['aesthetic']['std']:.4f}")
        print(f"Sharpness:       {aggregate_stats['sharpness']['mean']:.2f} ± {aggregate_stats['sharpness']['std']:.2f}")
    
    # Save results to JSON
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")
    
    return eval_results
def run_personalization_experiments_church(pipe, output_dir):
    """Runs the personalization experiments with multiple prompts and evaluates the results"""
    
    # Load evaluation models
    print("Loading evaluation models...")
    eval_models = load_evaluation_models()
    
    prompts = generate_personalization_prompts_for_SFM_SEEN()
    prompts_ = generate_personalization_prompts_for_SEEN()
    # Loop over each prompt
    eval_results = {}
    all_metrics = []
    
    for idx, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {idx+1}/{len(prompts)}: {prompt[:50]}...")
        
        input_images = ["/nvme-data/Komal/documents/omni_datasets/sfm/images/frame7.png"]
        
        # Generate output image using the pipe
        images = pipe(
            prompt=prompt,
            input_images=input_images,
            height=1024,
            width=1024,
            guidance_scale=2.5,
            img_guidance_scale=1.6,
            seed=0
        )
        
        # Save the generated image
        output_path = os.path.join(output_dir, f'personalized_image_{idx}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        images[0].save(output_path)
        print(f"Output saved to {output_path}")
        
        # Evaluate the generated image
        print("Computing evaluation metrics...")
        
        try:
            # CLIP-I Score (Image-to-Image similarity)
            clip_i = compute_clip_i_score(input_images[0], output_path, eval_models)
            
            # CLIP-T Score (Text-to-Image similarity)
            clip_t = compute_clip_t_score(prompts_[idx], output_path, eval_models)
            
            # DINV-1 Score (Perceptual similarity)
            dinv1 = compute_dinv1_score(input_images[0], output_path, eval_models)
            
            # Image quality metrics
            quality_metrics = compute_image_quality_metrics(output_path)
            
            # Aesthetic score
            aesthetic_metrics = compute_aesthetic_score(output_path, eval_models)
            
            # Compile results for this prompt
            prompt_results = {
                'prompt_idx': idx,
                'prompt': prompt,
                'output_path': output_path,
                'clip_i_score': clip_i,
                'clip_t_score': clip_t,
                'dinv1_score': dinv1,
                'quality_metrics': quality_metrics,
                'aesthetic_metrics': aesthetic_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            eval_results[f'prompt_{idx}'] = prompt_results
            all_metrics.append(prompt_results)
            
            print(f"  CLIP-I: {clip_i:.4f}")
            print(f"  CLIP-T: {clip_t:.4f}")
            print(f"  DINV-1: {dinv1:.4f}")
            print(f"  Aesthetic: {aesthetic_metrics['aesthetic_score']:.4f}")
            print(f"  Sharpness: {quality_metrics['sharpness']:.2f}")
            
        except Exception as e:
            print(f"Error computing metrics for prompt {idx}: {e}")
            eval_results[f'prompt_{idx}'] = {
                'prompt_idx': idx,
                'prompt': prompt,
                'output_path': output_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Compute aggregate statistics
    if all_metrics:
        print("\n" + "="*50)
        print("AGGREGATE RESULTS")
        print("="*50)
        
        clip_i_scores = [m['clip_i_score'] for m in all_metrics if 'clip_i_score' in m]
        clip_t_scores = [m['clip_t_score'] for m in all_metrics if 'clip_t_score' in m]
        dinv1_scores = [m['dinv1_score'] for m in all_metrics if 'dinv1_score' in m]
        aesthetic_scores = [m['aesthetic_metrics']['aesthetic_score'] for m in all_metrics if 'aesthetic_metrics' in m]
        sharpness_scores = [m['quality_metrics']['sharpness'] for m in all_metrics if 'quality_metrics' in m]
        
        aggregate_stats = {
            'clip_i': {
                'mean': np.mean(clip_i_scores),
                'std': np.std(clip_i_scores),
                'min': np.min(clip_i_scores),
                'max': np.max(clip_i_scores)
            },
            'clip_t': {
                'mean': np.mean(clip_t_scores),
                'std': np.std(clip_t_scores),
                'min': np.min(clip_t_scores),
                'max': np.max(clip_t_scores)
            },
            'dinv1': {
                'mean': np.mean(dinv1_scores),
                'std': np.std(dinv1_scores),
                'min': np.min(dinv1_scores),
                'max': np.max(dinv1_scores)
            },
            'aesthetic': {
                'mean': np.mean(aesthetic_scores),
                'std': np.std(aesthetic_scores),
                'min': np.min(aesthetic_scores),
                'max': np.max(aesthetic_scores)
            },
            'sharpness': {
                'mean': np.mean(sharpness_scores),
                'std': np.std(sharpness_scores),
                'min': np.min(sharpness_scores),
                'max': np.max(sharpness_scores)
            }
        }
        
        eval_results['aggregate_statistics'] = aggregate_stats
        
        # Print summary
        print(f"CLIP-I Score:    {aggregate_stats['clip_i']['mean']:.4f} ± {aggregate_stats['clip_i']['std']:.4f}")
        print(f"CLIP-T Score:    {aggregate_stats['clip_t']['mean']:.4f} ± {aggregate_stats['clip_t']['std']:.4f}")
        print(f"DINV-1 Score:    {aggregate_stats['dinv1']['mean']:.4f} ± {aggregate_stats['dinv1']['std']:.4f}")
        print(f"Aesthetic Score: {aggregate_stats['aesthetic']['mean']:.4f} ± {aggregate_stats['aesthetic']['std']:.4f}")
        print(f"Sharpness:       {aggregate_stats['sharpness']['mean']:.2f} ± {aggregate_stats['sharpness']['std']:.2f}")
    
    # Save results to JSON
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")
    
    return eval_results

def test_genration(output_images):
    """Test the generation process with multiple prompts and evaluate the results"""
    input_images = ["/nvme-data/Komal/documents/omni_datasets/sfm/images/frame7.png"]
    prompts = generate_personalization_prompts_for_SEEN()
    list_images = os.listdir(output_images)
    # prompts = generate_personalization_prompts_for_dog_uncond()
    eval_models = load_evaluation_models()
    # assert len(list_images) == len(prompts), "Number of images and prompts must match"
    eval_results = {}
    all_metrics = []
    for idx, image_name in enumerate(list_images):
        print(f"\nProcessing image {idx+1}/{len(list_images)}: {image_name}...")
        output_path = os.path.join(output_images, image_name)
        prompt = prompts[idx]
        #Taking 
        try:
            # CLIP-I Score (Image-to-Image similarity)
            clip_i = compute_clip_i_score(input_images[0], output_path, eval_models)
            
            # CLIP-T Score (Text-to-Image similarity)
            clip_t = compute_clip_t_score(prompt, output_path, eval_models)
            
            # DINV-1 Score (Perceptual similarity)
            dinv1 = compute_dinv1_score(input_images[0], output_path, eval_models)
            
            # Image quality metrics
            quality_metrics = compute_image_quality_metrics(output_path)
            
            # Aesthetic score
            aesthetic_metrics = compute_aesthetic_score(output_path, eval_models)
            
            # Compile results for this prompt
            prompt_results = {
                'prompt_idx': idx,
                'prompt': prompt,
                'output_path': output_path,
                'clip_i_score': clip_i,
                'clip_t_score': clip_t,
                'dinv1_score': dinv1,
                'quality_metrics': quality_metrics,
                'aesthetic_metrics': aesthetic_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            eval_results[f'prompt_{idx}'] = prompt_results
            all_metrics.append(prompt_results)
            
            print(f"  CLIP-I: {clip_i:.4f}")
            print(f"  CLIP-T: {clip_t:.4f}")
            print(f"  DINV-1: {dinv1:.4f}")
            print(f"  Aesthetic: {aesthetic_metrics['aesthetic_score']:.4f}")
            print(f"  Sharpness: {quality_metrics['sharpness']:.2f}")
            
        except Exception as e:
            print(f"Error computing metrics for prompt {idx}: {e}")
            eval_results[f'prompt_{idx}'] = {
                'prompt_idx': idx,
                'prompt': prompt,
                'output_path': output_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Compute aggregate statistics
    if all_metrics:
        print("\n" + "="*50)
        print("AGGREGATE RESULTS")
        print("="*50)
        
        clip_i_scores = [m['clip_i_score'] for m in all_metrics if 'clip_i_score' in m]
        clip_t_scores = [m['clip_t_score'] for m in all_metrics if 'clip_t_score' in m]
        dinv1_scores = [m['dinv1_score'] for m in all_metrics if 'dinv1_score' in m]
        aesthetic_scores = [m['aesthetic_metrics']['aesthetic_score'] for m in all_metrics if 'aesthetic_metrics' in m]
        sharpness_scores = [m['quality_metrics']['sharpness'] for m in all_metrics if 'quality_metrics' in m]
        
        aggregate_stats = {
            'clip_i': {
                'mean': np.mean(clip_i_scores),
                'std': np.std(clip_i_scores),
                'min': np.min(clip_i_scores),
                'max': np.max(clip_i_scores)
            },
            'clip_t': {
                'mean': np.mean(clip_t_scores),
                'std': np.std(clip_t_scores),
                'min': np.min(clip_t_scores),
                'max': np.max(clip_t_scores)
            },
            'dinv1': {
                'mean': np.mean(dinv1_scores),
                'std': np.std(dinv1_scores),
                'min': np.min(dinv1_scores),
                'max': np.max(dinv1_scores)
            },
            'aesthetic': {
                'mean': np.mean(aesthetic_scores),
                'std': np.std(aesthetic_scores),
                'min': np.min(aesthetic_scores),
                'max': np.max(aesthetic_scores)
            },
            'sharpness': {
                'mean': np.mean(sharpness_scores),
                'std': np.std(sharpness_scores),
                'min': np.min(sharpness_scores),
                'max': np.max(sharpness_scores)
            }
        }
        
        eval_results['aggregate_statistics'] = aggregate_stats
        
        # Print summary
        print(f"CLIP-I Score:    {aggregate_stats['clip_i']['mean']:.4f} ± {aggregate_stats['clip_i']['std']:.4f}")
        print(f"CLIP-T Score:    {aggregate_stats['clip_t']['mean']:.4f} ± {aggregate_stats['clip_t']['std']:.4f}")
        print(f"DINV-1 Score:    {aggregate_stats['dinv1']['mean']:.4f} ± {aggregate_stats['dinv1']['std']:.4f}")
        print(f"Aesthetic Score: {aggregate_stats['aesthetic']['mean']:.4f} ± {aggregate_stats['aesthetic']['std']:.4f}")
        print(f"Sharpness:       {aggregate_stats['sharpness']['mean']:.2f} ± {aggregate_stats['sharpness']['std']:.2f}")
    
    # Save results to JSON
    results_path = os.path.join(output_images, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")



import os
from tqdm import tqdm

def run_scene_experiments(pipe, output_dir):
    # List of 10 diverse scenarios based on the original prompt
    prompts = [
        "a photo of a white desk with a computer and a chair, with a lamp <img><|image_1|></img>.",
        "a photo of a white desk with a computer and a chair, with books on the desk <img><|image_1|></img>.",
        "a photo of a white desk with a computer and a chair, with a coffee cup on the desk <img><|image_1|></img>.",
        "a photo of a white desk with a computer and a chair, with a plant beside the computer <img><|image_1|></img>.",
        "a photo of a white desk with a computer and a chair, with papers scattered on the desk <img><|image_1|></img>.",
        "a photo of a white desk with a computer and a chair, with a clock on the wall <img><|image_1|></img>.",
        "a photo of a white desk with a computer and a chair, with a window showing the outdoors <img><|image_1|></img>.",
        "a photo of a white desk with a computer and a chair, with a stack of books on one side <img><|image_1|></img>.",
        "a photo of a white desk with a computer and a chair, with a smartphone on the desk <img><|image_1|></img>.",
        "a photo of a white desk with a computer and a chair, with a notepad and pen next to the keyboard <img><|image_1|></img>."
    ]
    
    input_images = ["/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/Scenes/easy/office_002/rgb_014.jpg"]  # Example input image
    
    # Loop through each of the 10 scenarios
    for idx, prompt in enumerate(prompts):
        images = pipe(
            prompt=prompt,
            input_images=input_images,
            height=1024,
            width=1024,
            guidance_scale=2.5,
            img_guidance_scale=1.6,
            seed=0
        )
        
        # Save the output for each scenario
        output_path = os.path.join(output_dir, f'scene_{idx+1:03d}', f'{prompt.split()[0]}_output.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        images[0].save(output_path)
        print(f"Output saved to {output_path}")

def main():
    args = setup_args()
    from modules.deft import add_knowledge_injection_methods
    from peft import LoraConfig
    from peft import LoraConfig
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"Using GPU: {args.gpu}")
    
    # Create output directories
    ensure_output_dir(args.output_dir)
    print(f"Outputs will be saved to: {args.output_dir}")
    
    # Import and load model
    print("Loading OmniGen model...")
    from OmniGen import OmniGenPipeline
    # pipe = OmniGenPipeline.from_pretrained("/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/sfm_based/checkpoints/0003200")
    # Loading original
    # adapter_path = f"/nvme-data/Komal/documents/results/DREAMBOOT/dog6/{args.decomposition_method}/checkpoints/0000400"
    # adapter_path = "/nvme-data/Komal/documents/results/para_Komal__finetune_lora_sks_dog/checkpoints/0000600/"
    print(f"Loading adapter weights from: {args.adapter_path}")
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
    print("Model loaded successfully!")
    if args.adapter_type == "para":
        transformer_lora_config = LoraConfigExtended(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["qkv_proj", "o_proj"],
            decomposition_method=args.decomposition_method
        )

        # pipe.model = load_para_rank_adapter(pipe.model, adapter_path, transformer_lora_config)
    elif args.adapter_type == "deft":
        print("Using deft rank adapter loading method")
        new_model = add_knowledge_injection_methods(pipe.model)
        # breakpoint()
        pipe.model = new_model.load_knowledge_injection_adapter(new_model, args.adapter_path)
    elif args.adapter_type == "lora":
        print("Using lora adapter loading method")
        pipe.merge_lora(args.adapter_path)

    # run_personalization_experiments(pipe, args.output_dir)
    run_personalization_experiments_omkar(pipe, args.output_dir)
    # run_personalization_experiments_church(pipe, args.output_dir)

    # run_scene_experiments(pipe, args.output_dir)

    print("\n=== All experiments completed successfully! ===")
    print(f"Results are saved in {args.output_dir}")

if __name__ == "__main__":
    main()
    # test_genration("/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/sfm_seen/para_for_prompting")
    # print("Testing generation with multiple prompts...")
    # print("Runing generation tests for LRMF...")
    # test_genration("/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/dreamboot/dog6/lrmf")
    # print("Runing generation tests for NMF...")
    # test_genration("/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/dreamboot/dog6/nmf")
    # print("Runing generation tests for QR...")
    # test_genration("/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/dreamboot/dog6/qr")
    # print("Runing generation tests for TSVD...")
    # test_genration("/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/dreamboot/dog6/tsvd")
# CUDA_VISIBLE_DEVICES=0 python evaluations/eval_personalization.py --output_dir /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/abalations_rebuttals/DEFT-no_decompo/2000 --decomposition_method "qr" --adapter_path /nvme-data/Komal/documents/results/Abalation/checkpoints/0002000 --adapter_type 'deft'
# CUDA_VISIBLE_DEVICES=0 python evaluations/eval_personalization.py --output_dir /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/abalations_rebuttals/lora/8000 --decomposition_method None --adapter_path /nvme-data/Komal/documents/results/Abalation/lora/checkpoints/0008000/ --adapter_type 'lora'

# Computing evaluation metrics...
#   CLIP-I: 0.8501
#   CLIP-T: 0.3250
#   DINV-1: 0.3549
#   Aesthetic: 0.0202
#   Sharpness: 96.76

# CUDA_VISIBLE_DEVICES=0 python evaluations/eval_personalization.py --output_dir /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/abalations_rebuttals/lorapara/2000 --decomposition_method 'qr' --adapter_path /nvme-data/Komal/documents/results/Abalation/lorapara//checkpoints/0002000/ --adapter_type 'deft'

# CUDA_VISIBLE_DEVICES=2 python evaluations/eval_personalization.py --output_dir /home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/abalations_rebuttals/PreluP^T_lr_r8/900 --decomposition_method None --adapter_path /nvme-data/Komal/documents/results/Abalation/PreluP^T_lr_r8/checkpoints/0000900 --adapter_type 'deft'