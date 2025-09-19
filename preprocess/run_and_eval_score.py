import os
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import f1_score
import clip
from lpips import LPIPS
from skimage.metrics import structural_similarity as ssim
from scipy import linalg
from torchvision.models import inception_v3
from torch.nn import functional as F

class ImageEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Initialize CLIP model for CLIP score calculation
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # Initialize LPIPS model
        self.lpips_model = LPIPS(net='alex').to(device)
        
        # Initialize inception model for FID calculation
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception_model.eval()
        # Remove final classification layer
        self.inception_model.fc = torch.nn.Identity()
        
        # Standard image transform for inception model
        self.inception_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def calculate_fid(self, real_images_dir, generated_images_dir):
        """
        Calculate Fréchet Inception Distance between real and generated images
        """
        real_features = self._get_inception_features(real_images_dir)
        gen_features = self._get_inception_features(generated_images_dir)
        
        # Calculate mean and covariance
        mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = gen_features.mean(0), np.cov(gen_features, rowvar=False)
        
        # Calculate FID
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid
    
    def _get_inception_features(self, image_dir):
        """Extract features from inception model for a directory of images"""
        features = []
        for img_file in os.listdir(image_dir):
            if not img_file.endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.inception_transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature = self.inception_model(img_tensor).squeeze().cpu().numpy()
                features.append(feature)
        
        return np.array(features)
    
    def calculate_ssim(self, original_image_path, generated_image_path):
        """
        Calculate Structural Similarity Index between original and generated image
        """
        original = np.array(Image.open(original_image_path).convert('RGB'))
        generated = np.array(Image.open(generated_image_path).convert('RGB'))
        
        # Ensure images are same size
        if original.shape != generated.shape:
            generated_img = Image.open(generated_image_path).convert('RGB')
            generated_img = generated_img.resize((original.shape[1], original.shape[0]))
            generated = np.array(generated_img)
        
        # Calculate SSIM for each channel and average
        ssim_value = 0
        for channel in range(3):  # RGB channels
            ssim_value += ssim(original[:,:,channel], generated[:,:,channel], 
                               data_range=255, gaussian_weights=True, sigma=1.5)
        
        return ssim_value / 3.0
    
    def calculate_rmse(self, original_image_path, generated_image_path):
        """
        Calculate Root Mean Squared Error between original and generated image
        """
        original = np.array(Image.open(original_image_path).convert('RGB')).astype(np.float32) / 255.0
        generated = np.array(Image.open(generated_image_path).convert('RGB')).astype(np.float32) / 255.0
        
        # Ensure images are same size
        if original.shape != generated.shape:
            generated_img = Image.open(generated_image_path).convert('RGB')
            generated_img = generated_img.resize((original.shape[1], original.shape[0]))
            generated = np.array(generated_img).astype(np.float32) / 255.0
        
        mse = np.mean((original - generated) ** 2)
        rmse = np.sqrt(mse)
        return rmse
    
    def calculate_lpips(self, original_image_path, generated_image_path):
        """
        Calculate LPIPS (Learned Perceptual Image Patch Similarity) between original and generated image
        """
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        original = Image.open(original_image_path).convert('RGB')
        generated = Image.open(generated_image_path).convert('RGB')
        
        original_tensor = transform(original).unsqueeze(0).to(self.device)
        generated_tensor = transform(generated).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            lpips_value = self.lpips_model(original_tensor, generated_tensor).item()
        
        return lpips_value
    
    def calculate_clip_score(self, image_path, text_prompt):
        """
        Calculate CLIP score between image and text prompt
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_token = clip.tokenize([text_prompt]).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_token)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).item()
        
        return similarity
    
    def calculate_f1_for_canny(self, original_edge_path, generated_image_path):
        """
        Calculate F1 score for canny-to-image task by comparing edges
        """
        # Load the original edge image
        original_edge = np.array(Image.open(original_edge_path).convert('L')) > 127
        
        # Generate edge from the generated image
        generated = np.array(Image.open(generated_image_path).convert('RGB'))
        generated_gray = np.array(Image.open(generated_image_path).convert('L'))
        generated_edge = cv2.Canny(generated_gray, 100, 200) > 0
        
        # Resize if needed
        if original_edge.shape != generated_edge.shape:
            generated_edge = cv2.resize(generated_edge.astype(np.uint8), 
                                       (original_edge.shape[1], original_edge.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST) > 0
        
        # Calculate F1 score
        true_flat = original_edge.flatten()
        pred_flat = generated_edge.flatten()
        f1 = f1_score(true_flat, pred_flat)
        
        return f1
    
    def evaluate_experiment(self, experiment_name, real_images_dir, generated_image_path, 
                           original_image_path=None, text_prompt=None, canny_path=None):
        """
        Calculate all metrics for a single experiment
        """
        results = {
            'Experiment': experiment_name
        }
        
        # Calculate FID if real images directory is provided
        if real_images_dir and os.path.exists(real_images_dir):
            gen_dir = os.path.dirname(generated_image_path)
            # FID needs multiple images, so we'll use the directory containing the generated image
            results['FID'] = self.calculate_fid(real_images_dir, gen_dir)
        else:
            results['FID'] = None
        
        # Calculate SSIM and RMSE if original image is provided
        if original_image_path and os.path.exists(original_image_path):
            results['SSIM'] = self.calculate_ssim(original_image_path, generated_image_path)
            results['RMSE'] = self.calculate_rmse(original_image_path, generated_image_path)
            results['LPIPS'] = self.calculate_lpips(original_image_path, generated_image_path)
        else:
            results['SSIM'] = None
            results['RMSE'] = None
            results['LPIPS'] = None
        
        # Calculate CLIP score if text prompt is provided
        if text_prompt:
            results['CLIP_Score'] = self.calculate_clip_score(generated_image_path, text_prompt)
        else:
            results['CLIP_Score'] = None
        
        # Calculate F1 score for canny-to-image task if canny edge image is provided
        if canny_path and os.path.exists(canny_path):
            results['F1_Score'] = self.calculate_f1_for_canny(canny_path, generated_image_path)
        else:
            results['F1_Score'] = None
        
        return results
    
    def evaluate_all_experiments(self, experiments_config, output_dir, real_images_dir=None):
        """
        Evaluate all experiments and save results to Excel
        """
        all_results = []
        
        for exp in experiments_config["experiments"]:
            exp_name = exp["name"]
            generated_image_path = os.path.join(output_dir, f"{exp_name}.png")
            
            # Get original image path if available
            original_image_path = None
            if 'input_images' in exp and exp['input_images'] and len(exp['input_images']) > 0:
                original_image_path = exp['input_images'][0]
            
            # Canny path for F1 score (if it's a canny-to-image task)
            canny_path = None
            if 'canny_path' in exp:
                canny_path = exp['canny_path']
            
            # Get text prompt
            text_prompt = exp['prompt'] if 'prompt' in exp else None
            
            # Evaluate
            results = self.evaluate_experiment(
                exp_name, 
                real_images_dir, 
                generated_image_path,
                original_image_path,
                text_prompt,
                canny_path
            )
            
            all_results.append(results)
        
        # Create DataFrame and save to Excel
        results_df = pd.DataFrame(all_results)
        excel_path = os.path.join(output_dir, "evaluation_results.xlsx")
        results_df.to_excel(excel_path, index=False)
        
        print(f"Evaluation completed! Results saved to {excel_path}")
        return results_df

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Evaluate generated images')
#     parser.add_argument('--experiment_config', type=str, required=True, 
#                         help='Path to experiment configuration JSON file')
#     parser.add_argument('--output_dir', type=str, required=True,
#                         help='Directory containing generated images and where to save results')
#     parser.add_argument('--real_images_dir', type=str, default=None,
#                         help='Directory containing real images for FID calculation')
    
#     return parser.parse_args()

# def main():
#     args = parse_arguments()
    
#     # Load experiment configuration
#     with open(args.experiment_config, 'r') as f:
#         experiments_config = json.load(f)
    
#     # Create evaluator
#     evaluator = ImageEvaluator()
    
#     # Run evaluation
#     results_df = evaluator.evaluate_all_experiments(
#         experiments_config,
#         args.output_dir,
#         args.real_images_dir
#     )
    
#     # Print summary
#     print("\nEvaluation Summary:")
#     print(results_df.describe())

# if __name__ == "__main__":
#  # Import OpenCV for edge detection
#     main()

import json
import cv2 
import json
import os
import argparse
from PIL import Image
import torch
import numpy as np
import pandas as pd

def setup_args():
    parser = argparse.ArgumentParser(description='Run OmniGen experiments with evaluation')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--output_dir', type=str, default='/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/INSDeT/INJ?QR', help='Directory to save outputs')
    parser.add_argument('--max_image_size', type=int, default=1024, help='Maximum input image size')
    parser.add_argument('--decomposition_method', type=str, default='qr', help='Decomposition method to use')
    parser.add_argument('--adapter_path', type=str, default='/nvme-data/Komal/documents/results/InsDet-FULL/OBJ/inj/blip2full/qr/checkpoints/0002000/', help='Path to the addaptor model')
    parser.add_argument('--lora_rank', type=int, default=8, help='Rank for LoRA')
    parser.add_argument('--lora_alpha', type=int, default=8, help='Alpha for LoRA')
    parser.add_argument('--is_parainj', action='store_true', help='Use para injection')
    parser.add_argument('--is_para', action='store_true', help='Use Para')
    parser.add_argument('--real_images_dir', type=str, default=None, help='Directory with real images for FID calculation')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only on existing generated images')
    parser.add_argument('--experiment_config', type=str, default="/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/llama_generated_experiments.json", help='Path to experiment configuration JSON file')
    
    return parser.parse_args()

def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)

def _run_experiments(pipe, experiments, output_dir, max_input_image_size, image_root):
    """Helper function to run experiments"""
    os.makedirs(output_dir, exist_ok=True)
    
    for exp in experiments:
        print(f"\nRunning experiment: {exp['name']}")
        print(f"Prompt: {exp['prompt'][:100]}...")
        images_ = []
        
        if exp['input_images']:
            print("Input images:")
            for img_path in exp['input_images']:
                full_img_path = os.path.join(image_root, img_path)
                images_.append(full_img_path)
                print(f"- {full_img_path}")
        exp['input_images'] = images_

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

def run_evaluation(experiment_config, output_dir, real_images_dir=None):
    """Run evaluation on generated images"""
    print("\n=== Running Image Evaluation ===")
    evaluator = ImageEvaluator()
    
    # Load experiments from config
    with open(experiment_config, 'r') as f:
        experiments_config = json.load(f)
    
    # Evaluate images
    results_df = evaluator.evaluate_all_experiments(
        experiments_config,
        output_dir,
        real_images_dir
    )
    
    # Print summary statistics
    print("\nEvaluation Summary Statistics:")
    print(results_df.describe())
    
    # Also create a summary table with averages for each metric
    summary_df = pd.DataFrame({
        'Metric': ['FID', 'SSIM', 'RMSE', 'LPIPS', 'CLIP_Score', 'F1_Score'],
        'Average': [
            results_df['FID'].mean() if 'FID' in results_df.columns and not results_df['FID'].isnull().all() else 'N/A',
            results_df['SSIM'].mean() if 'SSIM' in results_df.columns and not results_df['SSIM'].isnull().all() else 'N/A',
            results_df['RMSE'].mean() if 'RMSE' in results_df.columns and not results_df['RMSE'].isnull().all() else 'N/A',
            results_df['LPIPS'].mean() if 'LPIPS' in results_df.columns and not results_df['LPIPS'].isnull().all() else 'N/A',
            results_df['CLIP_Score'].mean() if 'CLIP_Score' in results_df.columns and not results_df['CLIP_Score'].isnull().all() else 'N/A',
            results_df['F1_Score'].mean() if 'F1_Score' in results_df.columns and not results_df['F1_Score'].isnull().all() else 'N/A'
        ]
    })
    
    # Save summary to Excel
    summary_path = os.path.join(output_dir, "evaluation_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)
    print(f"Summary saved to {summary_path}")
    
    return results_df

def main():
    args = setup_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU: {args.gpu}")
    
    # Create output directories
    ensure_output_dir(args.output_dir)
    print(f"Outputs will be saved to: {args.output_dir}")
    
    # Image root directory
    image_root = "/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/"
    
    # Get experiment configuration file
    experiment_config = args.experiment_config
    if experiment_config is None:
        experiment_config = "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/llama_generated_experiments.json"
    
    # If eval_only is False, run the model to generate images first
    if not args.eval_only:
        # Import and load model
        print("Loading OmniGen model...")
        from OmniGen import OmniGenPipeline
        from modules.utils import LoraConfigExtended, load_para_rank_adapter
        from modules.pepara import make_para_rank_adapter
        from modules.prepainj import add_knowledge_injection_methods
        from peft import LoraConfig
        
        # Load experiments
        with open(experiment_config, 'r') as f:
            data = json.load(f)
            
        adapter_path = args.adapter_path
        print(f"Loading adapter weights from: {adapter_path}")
        pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
        print("Model loaded successfully!")
        
        # Apply appropriate adapter method
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
            pipe.model = new_model.load_knowledge_injection_adapter(new_model, args.adapter_path)
            print("Para Injection applied successfully!")
        
        # Run experiments to generate images
        _run_experiments(pipe, data["experiments"], args.output_dir, args.max_image_size, image_root)
        print("\n=== All experiments completed successfully! ===")
    else:
        print("Skipping image generation, running evaluation only...")
    
    # Run evaluation on generated images
    results_df = run_evaluation(experiment_config, args.output_dir, args.real_images_dir)
    
    print(f"\n=== All tasks completed! ===")
    print(f"Generated images and evaluation results are saved in {args.output_dir}")

if __name__ == "__main__":
    main()