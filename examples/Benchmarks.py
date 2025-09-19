import os
import argparse
import time
import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity
import clip

# Import evaluation metrics
from lpips import LPIPS
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class OmniGenExperiments:
    def __init__(self, pipe, args):
        self.pipe = pipe
        self.output_dir = args.output_dir
        self.max_image_size = args.max_image_size
        self.decomposition_method = args.decomposition_method
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cuda")
        self.lpips_model = LPIPS(net='alex').cuda()
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
    def evaluate_image(self, original_img_path, generated_img_path):
        """Evaluate image quality using multiple metrics"""
        # Load images
        original_img = Image.open(original_img_path)
        generated_img = Image.open(generated_img_path)
        
        # Convert to numpy for structural metrics
        original_np = np.array(original_img.resize((256, 256)).convert('RGB'))
        generated_np = np.array(generated_img.resize((256, 256)).convert('RGB'))
        
        # Calculate SSIM and PSNR
        ssim_value = ssim(original_np, generated_np, channel_axis=-1, multichannel=True)
        psnr_value = psnr(original_np, generated_np)
        
        # Calculate LPIPS (perceptual similarity)
        original_tensor = torch.from_numpy(original_np).permute(2, 0, 1).float().cuda() / 255.0
        generated_tensor = torch.from_numpy(generated_np).permute(2, 0, 1).float().cuda() / 255.0
        original_tensor = original_tensor.unsqueeze(0)
        generated_tensor = generated_tensor.unsqueeze(0)
        lpips_value = self.lpips_model(original_tensor, generated_tensor).item()
        
        # Calculate CLIP similarity
        original_clip = self.clip_preprocess(original_img).unsqueeze(0).to("cuda")
        generated_clip = self.clip_preprocess(generated_img).unsqueeze(0).to("cuda")
        
        with torch.no_grad():
            original_features = self.clip_model.encode_image(original_clip)
            generated_features = self.clip_model.encode_image(generated_clip)
            
        clip_similarity = cosine_similarity(
            original_features.cpu().numpy(), 
            generated_features.cpu().numpy()
        )[0][0]
        
        return {
            "ssim": float(ssim_value),
            "psnr": float(psnr_value),
            "lpips": float(lpips_value),
            "clip_similarity": float(clip_similarity)
        }
    def run_all_experiments(self):
        """Run all experiments in sequence"""
        # 1. Identity Preservation Experiments
        self.run_identity_preservation_experiments()
        
        # 2. Multi-Concept Composition Experiments
        self.run_multi_concept_composition_experiments()
        
        # 3. Ablation Studies
        self.run_ablation_studies()
        
        # 4. Generalization Tests
        self.run_generalization_tests()
        
        # 5. Image Editing Experiments
        self.run_image_editing_experiments()
        
        # 6. Robustness Tests
        self.run_robustness_tests()

    def save_image(self, image, subdir, filename):
        """Save generated image to specified subdirectory"""
        output_path = os.path.join(self.output_dir, subdir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        return output_path



    def generate_and_evaluate(self, subdir, experiment_name, prompt, input_images, 
                             guidance_scale=2.5, img_guidance_scale=1.6, seed=0,
                             evaluate=True, reference_idx=0):
        """Generate an image and evaluate it against the reference"""
        print(f"\nRunning experiment: {experiment_name}")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        print("Input images:")
        for img_path in input_images:
            print(f"- {img_path}")
        
        # Generate image
        start_time = time.time()
        
        images = self.pipe(
            prompt=prompt,
            input_images=input_images,
            height=1024,
            width=1024,
            guidance_scale=guidance_scale,
            img_guidance_scale=img_guidance_scale,
            max_input_image_size=self.max_image_size,
            seed=seed
        )
        
        generation_time = time.time() - start_time
        
        # Save output
        output_path = self.save_image(images[0], subdir, f"{experiment_name}.png")
        print(f"Output saved to {output_path}")
        
        # Evaluate if requested
        results = {"generation_time": generation_time}
        
        if evaluate and input_images:
            reference_path = input_images[reference_idx]
            evaluation_metrics = self.evaluate_image(reference_path, output_path)
            results.update(evaluation_metrics)
            print("Evaluation metrics:")
            for metric, value in evaluation_metrics.items():
                print(f"- {metric}: {value:.4f}")
        
        return results

    def run_identity_preservation_experiments(self):
        """Run identity preservation experiments"""
        print("\n=== Running Identity Preservation Experiments ===")
        
        experiments_dir = "identity_preservation"
        evaluation_results = {}
        
        # 1. Single Object Identity Preservation
        experiments = [
            {
                "name": "single_object_park",
                "prompt": "A sks dog <img><|image_1|></img> playing at the park with a ball, bright sunny day",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 42}
            },
            {
                "name": "single_object_bookshelf",
                "prompt": "A sks dog <img><|image_1|></img> sitting in front of a bookshelf in a library",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 42}
            },
            {
                "name": "single_object_beach",
                "prompt": "A sks dog <img><|image_1|></img> running on a beach with waves crashing in the background",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 42}
            },
            # 2. Multiple Object Identity Preservation
            {
                "name": "multiple_objects_person_dog",
                "prompt": "A woman with a floral dress is sitting with a sks dog <img><|image_1|></img> in a garden. The woman is the one in <img><|image_2|></img>.",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg", 
                                "./imgs/test_cases/4.jpg"],
                "params": {"seed": 24}
            },
            # 3. Identity Preservation Across Different Poses
            {
                "name": "different_pose_sitting",
                "prompt": "A sks dog <img><|image_1|></img> sitting obediently in front of a fireplace",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 36}
            },
            {
                "name": "different_pose_running",
                "prompt": "A sks dog <img><|image_1|></img> running and leaping through tall grass",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 36}
            },
            # 4. Identity Preservation with Image Editing
            {
                "name": "image_editing_background",
                "prompt": "A sks dog <img><|image_1|></img> in the same pose but with a snow-covered mountain background",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 48}
            },
            # 5. Cross-domain Identity Consistency
            {
                "name": "cross_domain_painting",
                "prompt": "A Van Gogh style painting of a sks dog <img><|image_1|></img> in a field of sunflowers",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 60}
            },
            {
                "name": "cross_domain_cartoon",
                "prompt": "A Pixar-style 3D animation of a sks dog <img><|image_1|></img> with exaggerated features",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 60}
            }
        ]
        
        for exp in experiments:
            results = self.generate_and_evaluate(
                experiments_dir,
                exp["name"],
                exp["prompt"],
                exp["input_images"],
                seed=exp["params"].get("seed", 0),
                guidance_scale=exp["params"].get("guidance_scale", 2.5),
                img_guidance_scale=exp["params"].get("img_guidance_scale", 1.6)
            )
            evaluation_results[exp["name"]] = results
        
        # Save evaluation results
        eval_path = os.path.join(self.output_dir, experiments_dir, "evaluation_results.json")
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Evaluation results saved to {eval_path}")

    def run_multi_concept_composition_experiments(self):
        """Run multi-concept composition experiments"""
        print("\n=== Running Multi-Concept Composition Experiments ===")
        
        experiments_dir = "multi_concept_composition"
        evaluation_results = {}
        
        experiments = [
            # 1. Two Identity Interaction
            {
                "name": "two_identity_dog_bird",
                "prompt": "A sks dog <img><|image_1|></img> and a beautiful blue bird sitting together in a garden",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 42}
            },
            # 2. Style and Identity Composition
            {
                "name": "style_identity_van_gogh",
                "prompt": "A sks dog <img><|image_1|></img> painted in the style of Van Gogh",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 24}
            },
            # 3. Multiple Identity Composition
            {
                "name": "multiple_identity_dog_cat_person",
                "prompt": "A sks dog <img><|image_1|></img>, a ginger cat <img><|image_2|></img>, and a person <img><|image_3|></img> sitting on a park bench together",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg", 
                                "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/concepts/cat/01.jpg",
                                "./imgs/test_cases/Komal.jpeg"],
                "params": {"seed": 36}
            },
            # 4. Interaction Between Multiple Subjects
            {
                "name": "interaction_teaching_dog",
                "prompt": "A man <img><|image_1|></img> and a woman <img><|image_2|></img> teaching a sks dog <img><|image_3|></img> to perform a trick in a living room",
                "input_images": ["./imgs/test_cases/turing.png", 
                                "./imgs/test_cases/4.jpg", 
                                "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 48}
            },
            # 5. Complex Scene with Mixed Concepts
            {
                "name": "complex_scene_desert",
                "prompt": "A sks dog <img><|image_1|></img> and a woman <img><|image_2|></img> in a desert landscape, the woman is giving water to the dog",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg", 
                                "./imgs/test_cases/Amanda.jpg"],
                "params": {"seed": 60}
            },
            # 6. Cross-object and Cross-style Composition
            {
                "name": "cross_object_style_cyberpunk",
                "prompt": "A futuristic sks dog <img><|image_1|></img> in a cyberpunk city with neon lights and flying cars",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 72}
            }
        ]
        
        for exp in experiments:
            results = self.generate_and_evaluate(
                experiments_dir,
                exp["name"],
                exp["prompt"],
                exp["input_images"],
                seed=exp["params"].get("seed", 0),
                guidance_scale=exp["params"].get("guidance_scale", 2.5),
                img_guidance_scale=exp["params"].get("img_guidance_scale", 1.6)
            )
            evaluation_results[exp["name"]] = results
        
        # Save evaluation results
        eval_path = os.path.join(self.output_dir, experiments_dir, "evaluation_results.json")
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Evaluation results saved to {eval_path}")

    def run_ablation_studies(self):
        """Run ablation studies for parameter efficiency"""
        print("\n=== Running Ablation Studies (Parameter Efficiency) ===")
        
        # Note: This is a placeholder for the actual ablation study
        # In a real implementation, you would need to reload the model with different rank configurations
        print("Note: Ablation studies require reloading models with different rank configurations.")
        print("This would typically be implemented as a separate script that calls this experiment runner with different model configurations.")
        
        # Example of what results collection might look like:
        ablation_results = {
            "rank_comparison": {
                "description": "Comparison of different ranks for parameter-efficient fine-tuning",
                "test_prompt": "A sks dog <img><|image_1|></img> playing in a meadow",
                "reference_image": "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg",
                "results": {
                    "rank_2": {"time": "placeholder", "metrics": "placeholder"},
                    "rank_4": {"time": "placeholder", "metrics": "placeholder"},
                    "rank_8": {"time": "placeholder", "metrics": "placeholder"},
                    "rank_16": {"time": "placeholder", "metrics": "placeholder"}
                }
            }
        }
        
        # Save placeholder results
        ablation_dir = "ablation_studies"
        os.makedirs(os.path.join(self.output_dir, ablation_dir), exist_ok=True)
        ablation_path = os.path.join(self.output_dir, ablation_dir, "rank_ablation_placeholder.json")
        with open(ablation_path, 'w') as f:
            json.dump(ablation_results, f, indent=4)
        print(f"Placeholder ablation results saved to {ablation_path}")

    def run_generalization_tests(self):
        """Run tests for generalization to unseen prompts"""
        print("\n=== Running Generalization Tests ===")
        
        experiments_dir = "generalization_tests"
        evaluation_results = {}
        
        # Define novel, unseen prompts
        experiments = [
            {
                "name": "unseen_prompt_abstract",
                "prompt": "A sks dog <img><|image_1|></img> swimming in a waterfall on Mars, with Earth visible in the sky",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 42}
            },
            {
                "name": "unseen_prompt_fantasy",
                "prompt": "A sks dog <img><|image_1|></img> with magical powers shooting beams of light from its eyes in an enchanted forest",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 24}
            },
            {
                "name": "unseen_prompt_historical",
                "prompt": "A sks dog <img><|image_1|></img> as a royal pet in medieval times, wearing a small crown and royal robe",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 36}
            },
            {
                "name": "unseen_prompt_futuristic",
                "prompt": "A sks dog <img><|image_1|></img> piloting a flying car in a futuristic city with hovering buildings",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 48}
            }
        ]
        
        for exp in experiments:
            results = self.generate_and_evaluate(
                experiments_dir,
                exp["name"],
                exp["prompt"],
                exp["input_images"],
                seed=exp["params"].get("seed", 0),
                guidance_scale=exp["params"].get("guidance_scale", 2.5),
                img_guidance_scale=exp["params"].get("img_guidance_scale", 1.6)
            )
            evaluation_results[exp["name"]] = results
        
        # Save evaluation results
        eval_path = os.path.join(self.output_dir, experiments_dir, "evaluation_results.json")
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Evaluation results saved to {eval_path}")

    def run_image_editing_experiments(self):
        """Run one-shot learning image editing experiments"""
        print("\n=== Running Image Editing (One-Shot Learning) Experiments ===")
        
        experiments_dir = "image_editing"
        evaluation_results = {}
        
        experiments = [
            {
                "name": "background_editing_park",
                "prompt": "Change the background of <img><|image_1|></img> to a beautiful park with green trees and a pond, keep the dog's identity exactly the same",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 42}
            },
            {
                "name": "background_editing_beach",
                "prompt": "Change the background of <img><|image_1|></img> to a sunny beach with blue sky and ocean waves, keep the dog's identity exactly the same",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 24}
            },
            {
                "name": "background_editing_city",
                "prompt": "Change the background of <img><|image_1|></img> to a busy city street with buildings and cars, keep the dog's identity exactly the same",
                "input_images": ["/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"],
                "params": {"seed": 36}
            }
        ]
        
        for exp in experiments:
            results = self.generate_and_evaluate(
                experiments_dir,
                exp["name"],
                exp["prompt"],
                exp["input_images"],
                seed=exp["params"].get("seed", 0),
                guidance_scale=exp["params"].get("guidance_scale", 2.5),
                img_guidance_scale=exp["params"].get("img_guidance_scale", 1.6)
            )
            evaluation_results[exp["name"]] = results
        
        # Save evaluation results
        eval_path = os.path.join(self.output_dir, experiments_dir, "evaluation_results.json")
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Evaluation results saved to {eval_path}")

    def run_robustness_tests(self):
        """Run robustness experiments with various perturbations"""
        print("\n=== Running Robustness Tests ===")
        
        experiments_dir = "robustness_tests"
        evaluation_results = {}
        
        # Load and create noisy versions of the reference image
        reference_img_path = "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/toy_data/images/dog4.jpeg"
        reference_img = Image.open(reference_img_path)
        
        # Create noisy versions with different levels of noise
        noise_levels = [0.05, 0.1, 0.2]
        noisy_images = []
        
        for noise_level in noise_levels:
            img_np = np.array(reference_img).astype(np.float32)
            noise = np.random.normal(0, noise_level * 255, img_np.shape)
            noisy_img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            noisy_img = Image.fromarray(noisy_img_np)
            
            # Save the noisy image
            noisy_img_path = os.path.join(self.output_dir, experiments_dir, f"noisy_input_{noise_level}.png")
            os.makedirs(os.path.dirname(noisy_img_path), exist_ok=True)
            noisy_img.save(noisy_img_path)
            noisy_images.append(noisy_img_path)
        
        # Run experiments with noisy images
        for i, noisy_img_path in enumerate(noisy_images):
            noise_level = noise_levels[i]
            exp_name = f"robustness_noise_{noise_level}"
            
            prompt = f"A sks dog <img><|image_1|></img> sitting in a garden with flowers"
            
            results = self.generate_and_evaluate(
                experiments_dir,
                exp_name,
                prompt,
                [noisy_img_path],
                seed=42,
                guidance_scale=2.5,
                img_guidance_scale=1.6,
                evaluate=True,
                reference_idx=0  # Compare with the noisy input
            )
            
            # Also evaluate against the original clean image
            clean_comparison = self.evaluate_image(reference_img_path, 
                                                 os.path.join(self.output_dir, experiments_dir, f"{exp_name}.png"))
            results["comparison_to_clean"] = clean_comparison
            
            evaluation_results[exp_name] = results
        
        # Save evaluation results
        eval_path = os.path.join(self.output_dir, experiments_dir, "evaluation_results.json")
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Evaluation results saved to {eval_path}")


def setup_args():
    parser = argparse.ArgumentParser(description='Run OmniGen comprehensive experiments')
    
    # Basic parameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--output_dir', type=str, default='./results/comprehensive_experiments', 
                        help='Directory to save outputs')
    parser.add_argument('--max_image_size', type=int, default=1024, 
                        help='Maximum input image size')
    parser.add_argument('--decomposition_method', type=str, default='qr', 
                        help='Decomposition method to use (qr, svd, lrmf)')
    
    # Experiment selection
    parser.add_argument('--run_identity_preservation', action='store_true', 
                        help='Run identity preservation experiments')
    parser.add_argument('--run_multi_concept', action='store_true', 
                        help='Run multi-concept composition experiments')
    parser.add_argument('--run_ablation', action='store_true', 
                        help='Run ablation studies')
    parser.add_argument('--run_generalization', action='store_true', 
                        help='Run generalization tests')
    parser.add_argument('--run_image_editing', action='store_true', 
                        help='Run image editing experiments')
    parser.add_argument('--run_robustness', action='store_true', 
                        help='Run robustness tests')
    parser.add_argument('--run_all', action='store_true', 
                        help='Run all experiments')
    
    # Adapter parameters
    parser.add_argument('--adapter_path', type=str, required=False,
                        default=None, help='Path to adapter weights')
    parser.add_argument('--lora_rank', type=int, default=8, 
                        help='Rank for LoRA configuration')
    
    return parser.parse_args()

def main():
    args = setup_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU: {args.gpu}")
    
    # Import necessary modules
    print("Loading required modules...")
    from modules.utils import LoraConfigExtended
    try:
        from modules.utils import load_para_rank_adapter
        para_rank_available = True
    except ImportError:
        print("Warning: PaRa rank adapter not available, falling back to standard LoRA")
        para_rank_available = False
    
    # Import and load model
    print("Loading OmniGen model...")
    from OmniGen import OmniGenPipeline
    
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
    print("Base model loaded successfully!")
    
    # Load adapter weights if specified
    if args.adapter_path:
        print(f"Loading adapter weights from: {args.adapter_path}")
        
        # Configure adapter based on availability
        if para_rank_available:
            transformer_lora_config = LoraConfigExtended(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,
                init_lora_weights="gaussian",
                target_modules=["qkv_proj", "o_proj"],
                decomposition_method=args.decomposition_method
            )
            pipe.model = load_para_rank_adapter(pipe.model, args.adapter_path, transformer_lora_config)
        else:
            pipe.merge_lora(args.adapter_path)
        
        print("Adapter weights loaded successfully!")
    
    # Initialize experiments runner
    runner = OmniGenExperiments(pipe, args)
    
    # Run selected experiments
    if args.run_all:
        runner.run_all_experiments()
    else:
        if args.run_identity_preservation:
            runner.run_identity_preservation_experiments()
        
        if args.run_multi_concept:
            runner.run_multi_concept_composition_experiments()
        
        if args.run_ablation:
            runner.run_ablation_studies()
        
        if args.run_generalization:
            runner.run_generalization_tests()
        
        if args.run_image_editing:
            runner.run_image_editing_experiments()
        
        if args.run_robustness:
            runner.run_robustness_tests()
    
    print("\n=== All experiments completed successfully! ===")
    print(f"Results are saved in {args.output_dir}")

if __name__ == "__main__":
    main()

# The above code is a comprehensive script for running various experiments with the OmniGen model.
