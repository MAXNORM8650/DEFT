import os
import argparse
from PIL import Image
from pathlib import Path

def setup_args():
    parser = argparse.ArgumentParser(description='Run OmniGen experiments')
    parser.add_argument('--gpu', type=str, default='2', help='GPU ID to use')
    parser.add_argument('--output_dir', type=str, default='./results/lays_ob', help='Directory to save outputs')
    parser.add_argument('--max_image_size', type=int, default=1024, help='Maximum input image size')
    return parser.parse_args()

def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    for subdir in ['text2img', 'img2img', 'identity_preservation', 'img_conditioning']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

def run_text2img_experiments(pipe, output_dir):
    """Run text-to-image generation experiments"""
    print("\n=== Running Text-to-Image Experiments ===")
    
    prompts = [
        "a photo of sks dog <img><|image_1|></img> in a lush green field, the dog is playing with a ball, the sun is shining brightly in the background",
        "A photo of sks dog <img><|image_1|></img> in beach, the dog is playing with a ball, the sun is setting in the background, waves are crashing on the shore",
        "A photo of sks dog <img><|image_1|></img> in a snowy landscape, the dog is playing with a snowball, snowflakes are falling gently from the sky",
        "A photo of sks dog <img><|image_1|></img> in a cityscape, the dog is sitting on a sidewalk, skyscrapers are towering in the background, people walking by",
        "A photo of sks dog <img><|image_1|></img> in a garden, the dog is sniffing flowers, butterflies are fluttering around, a small pond is visible in the background",
        "A photo of sks dog <img><|image_1|></img> in a forest, the dog is running through the trees, sunlight filtering through the leaves, a small stream is flowing nearby",
    ]
    input_images = ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"]  # Example input image
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
        
        output_path = os.path.join(output_dir, 'text2img_cond', f'output_{i}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        images[0].save(output_path)
        print(f"Image saved to {output_path}")
def run_levi_experiments(pipe, output_dir):
    """Run Levi image generation experiments"""
    print("\n=== Running Levi Image Generation Experiments ===")
    
    # Example Levi image generation experiment
    prompt = "A photo of sks komal <img><|image_1|></img> sitting in the same posture as levi in attack on titan, levi is sitting in the same posture as komal in <img><|image_2|></img>."
    input_images = ["./imgs/test_cases/Komal.jpeg", "./imgs/test_cases/levi.jpg"]  # Example input image for Levi
    
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=1024,
        width=1024,
        guidance_scale=2.5,
        img_guidance_scale=1.6,
        seed=0
    )

    
    # Save output
    output_path = os.path.join(output_dir, 'levi', 'levi_output.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(output_path)
    print(f"Output saved to {output_path}")

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
    
    experiments = [
        {
            "name": "single_object",
            "prompt": "A man in a black shirt is reading a book. The man is the right man in <img><|image_1|></img>.",
            "input_images": ["./imgs/test_cases/two_man.jpg"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.6,
                "seed": 0
            }
        },
        {
            "name": "multiple_objects",
            "prompt": "A man is sitting in the library reading a book, while a woman wearing white shirt next to him is wearing headphone. The man who is reading is the one wearing red sweater in <img><|image_1|></img>. The woman wearing headphones is the right woman wearing suit in <img><|image_2|></img>.",
            "input_images": ["./imgs/test_cases/turing.png", "./imgs/test_cases/lecun.png"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.8,
                "seed": 2
            }
        },
        {
            "name": "Komal_bookshelf_scene",
            "prompt": "A man and a short-haired woman with a wrinkled face are standing in front of a bookshelf in a library. The man is the man in the middle of <img><|image_1|></img>, and the woman is oldest woman in <img><|image_2|></img>",
            "input_images": ["./imgs/test_cases/Komal.jpeg", "./imgs/test_cases/2.jpg"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.6,
                "seed": 60
            }
        },
        {
            "name": "classroom_scene",
            "prompt": "A man and a woman are sitting at a classroom desk. The man is the man with yellow hair in <img><|image_1|></img>. The woman is the woman on the left of <img><|image_2|></img>",
            "input_images": ["./imgs/test_cases/3.jpg", "./imgs/test_cases/4.jpg"],
            "params": {
                "guidance_scale": 2.5,
                "img_guidance_scale": 1.6,
                "seed": 66
            }
        },
        {
            "name": "fashion_scene",
            "prompt": "A woman is walking down the street, wearing a white long-sleeve blouse with lace details on the sleeves, paired with a blue pleated skirt. The woman is <img><|image_1|></img>. The long-sleeve blouse and a pleated skirt are <img><|image_2|></img>.",
            "input_images": ["./imgs/demo_cases/emma.jpeg", "./imgs/demo_cases/dress.jpg"],
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
    prompt = "A photo of sks lays chip <img><|image_1|></img> on the image showcases <img><|image_2|></img>."
    input_images = ["/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/Scenes/easy/leisure_zone_001/rgb_000.jpg", "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/table_scheen/schene1.jpeg"]  # Example input image for lays chip
    
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
import random   

def main():
    args = setup_args()
    from modules.utils import load_para_rank_model
    from modules.pepara import make_para_rank_adapter
    from modules.merging import merge_para_models_direct
    from modules.seqmerging import merge_para_models_sequential
    from peft import LoraConfig
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    print(f"Using GPU: {args.gpu}")
    
    # Create output directories
    ensure_output_dir(args.output_dir)
    print(f"Outputs will be saved to: {args.output_dir}")
    
    # Import and load model
    print("Loading OmniGen model...")
    from OmniGen import OmniGenPipeline
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
    print("Model loaded successfully!")
    adapter_path = "/nvme-data/Komal/documents/results/lays_chip_tube_auburn/checkpoints/0001200/"
    print(f"Loading adapter weights from: {adapter_path}")
    transformer_lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        # target_modules=["qkv_proj", "o_proj"],
    )
    # Merging PaRa models 
    model1_path = "/nvme-data/Komal/documents/results/lays_chip_tube_auburn/checkpoints/0001200/"
    model2_path = "/nvme-data/Komal/documents/results/schene_lopara/checkpoints/0002000/"
    model_adp = "/nvme-data/Komal/documents/results/INJpara/SKSdog/checkpoints/0000400"

    
    # breakpoint()
    # pipe.model = merge_para_models_sequential(pipe.model, model1_path, model2_path)
    run_levi_experiments(pipe, args.output_dir)
    # direct_merged_model = merge_para_models_direct(pipe.model, model1_path, model2_path)

    # model = make_para_rank_adapter(pipe.model, transformer_lora_config)
    # pipe.model = load_para_rank_model(model, adapter_path, device='cuda')
    run_levi_experiments(pipe, args.output_dir)
    # Load the adapter weights into the pipeline
    # pipe.merge_lora(adapter_path)
    # Run experiments
    #Mask generation experiments
    # running_multiple_prompt_experiments(pipe, args.output_dir)
    # seq_output_dir = os.path.join(args.output_dir, 'seq_merged')
    # direct_output_dir = os.path.join(args.output_dir, 'direct_merged')
    # os.makedirs(seq_output_dir, exist_ok=True)
    # os.makedirs(direct_output_dir, exist_ok=True)
    # run_lays_chip_experiments(pipe, seq_output_dir)
    # run_lays_chip_experiments(pipe, args.output_dir)
    # run_lays_chip_experiments(direct_merged_model, direct_output_dir)
    # run_lays_experiments(pipe, args.output_dir)
    # run_objcet_mask_generation_experiments(pipe, args.output_dir)
    # run_depth_map_generation_experiments(pipe, args.output_dir)
    # run_object_detection_experiments(pipe, args.output_dir)
    # run_text2img_experiments(pipe, args.output_dir)
    # run_img2img_experiments(pipe, args.output_dir, args.max_image_size)
    # run_reasoning_experiments(pipe, args.output_dir)
    # run_identity_preservation_experiments(pipe, args.output_dir, args.max_image_size)
    # run_image_conditioning_experiments(pipe, args.output_dir)
    # run_Personalized_text2img_experiments(pipe, args.output_dir)
    # run_KPersonalized_text2img_experiments(pipe, args.output_dir)
    # run_Schene_Personalized_text2img_experiments(pipe, args.output_dir)
    print("\n=== All experiments completed successfully! ===")
    print(f"Results are saved in {args.output_dir}")

if __name__ == "__main__":
    main()