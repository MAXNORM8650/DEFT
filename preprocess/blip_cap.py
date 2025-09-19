import os
import argparse
import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
from tqdm import tqdm
import multiprocessing as mp
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ImageCaptioningDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        return self.image_paths[idx]

def caption_image(image_path, processor, model, device):
    """Generate caption for a single image"""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt").to(device)
        
        # Generate caption
        with torch.no_grad():
            out = model.generate(**inputs, max_length=75)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None

def process_batch(gpu_id, image_paths, src_dir, output_dir, batch_size=16):
    """Process a batch of images on a single GPU"""
    try:
        # Set device
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        logger.info(f"Worker {gpu_id} using device: {device}")
        
        # Load model for this worker
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
        
        # Process each image
        for image_path in tqdm(image_paths, desc=f"GPU {gpu_id}", position=gpu_id):
            try:
                # Create output directory structure
                rel_path = os.path.relpath(image_path, src_dir)
                output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.txt')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Skip if output already exists
                if os.path.exists(output_path):
                    continue
                
                # Generate caption
                caption = caption_image(image_path, processor, model, device)
                
                if caption:
                    # Save caption
                    with open(output_path, 'w') as f:
                        f.write(caption)
            
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Worker {gpu_id} encountered an error: {e}")
        return False

def get_all_image_files(src_dir):
    """Get all image files from the source directory"""
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(glob.glob(os.path.join(src_dir, '**', f'*.{ext}'), recursive=True))
    return image_files

def split_workload(image_files, num_gpus):
    """Split workload evenly across GPUs"""
    splits = []
    chunk_size = len(image_files) // num_gpus
    remainder = len(image_files) % num_gpus
    
    start_idx = 0
    for i in range(num_gpus):
        # Add one extra item to some chunks if the division has a remainder
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        splits.append(image_files[start_idx:end_idx])
        start_idx = end_idx
    
    return splits

def main():
    parser = argparse.ArgumentParser(description='BLIP Image Captioning with Multi-GPU Support')
    parser.add_argument('--src_dir', type=str, 
                        default='/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/eval/multi_gpu_test/omnigen-DEFT-32-2600/src_image_output',
                        help='Source directory containing images')
    parser.add_argument('--output_dir', type=str, 
                        default='/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/results/eval/multi_gpu_test/omnigen-DEFT-32-2600/blip_text',
                        help='Output directory for captions')
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(),
                        help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU instead.")
        args.num_gpus = 1
    else:
        # Get the number of available GPUs
        available_gpus = torch.cuda.device_count()
        if args.num_gpus > available_gpus:
            logger.warning(f"Requested {args.num_gpus} GPUs but only {available_gpus} are available. Using {available_gpus} GPUs.")
            args.num_gpus = available_gpus
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all image files
    logger.info("Finding all image files...")
    image_files = get_all_image_files(args.src_dir)
    logger.info(f"Found {len(image_files)} images to process")
    
    # Split workload across GPUs
    splits = split_workload(image_files, args.num_gpus)
    
    # Process images using multiprocessing with spawn method
    processes = []
    for gpu_id in range(args.num_gpus):
        p = mp.Process(
            target=process_batch,
            args=(gpu_id, splits[gpu_id], args.src_dir, args.output_dir, args.batch_size)
        )
        processes.append(p)
    
    # Start all processes
    for p in processes:
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    end_time = time.time()
    logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    # Set start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()