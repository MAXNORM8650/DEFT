import json
import math
import os
from typing import Literal, Dict, Any, Tuple, Optional, Union

import fire
import megfile
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
import torch.nn.functional as F
from accelerate import PartialState
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import BitImageProcessor, Dinov2Model

# For FID Score
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

# For SSIM
from skimage.metrics import structural_similarity as ssim

# For MUSIQ
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# For MAN-IQA
from transformers import AutoModelForSequenceClassification, AutoProcessor

from dreambench_plus.constants import LOCAL_FILES_ONLY, MODEL_ZOOS
from dreambench_plus.utils.comm import all_gather
from dreambench_plus.utils.image_utils import IMAGE_EXT, ImageType, load_image
from dreambench_plus.utils.loguru import logger

_DEFAULT_MODEL_V1: str = "dino_vits8"
_DEFAULT_MODEL_V2: str = MODEL_ZOOS["facebook/dinov2-small"]
_DEFAULT_TORCH_DTYPE: torch.dtype = torch.float32


class DinoScore:
    def __init__(
        self,
        model_or_name_path: str = _DEFAULT_MODEL_V1,
        torch_dtype: torch.dtype = _DEFAULT_TORCH_DTYPE,
        local_files_only: bool = LOCAL_FILES_ONLY,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype
        self.model = torch.hub.load("facebookresearch/dino:main", model_or_name_path).to(device, dtype=torch_dtype)
        self.model.eval()
        self.processor = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        if device is not None:
            self.device = device
            self.model = self.model.to(device)

        if dtype is not None:
            self.dtype = dtype
            self.model = self.model.to(dtype)

    @torch.no_grad()
    def get_image_features(self, image: ImageType | list[ImageType], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        inputs = [self.processor(i) for i in image]
        inputs = torch.stack(inputs).to(self.device, dtype=self.dtype)
        image_features = self.model(inputs)
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def dino_score(self, images1: ImageType | list[ImageType], images2: ImageType | list[ImageType]) -> tuple[float, int]:
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        assert len(images1) == len(images2), f"Number of images1 ({len(images1)}) and images2 {(len(images2))} should be same."

        images1_features = self.get_image_features(images1, norm=True)
        images2_features = self.get_image_features(images2, norm=True)
        # cosine similarity between feature vectors
        score = 100 * (images1_features * images2_features).sum(axis=-1)
        return score.sum(0).float(), len(images1)


class Dinov2Score(DinoScore):
    # NOTE: noqa, in version 1, the performance of the official repository and HuggingFace is inconsistent.
    def __init__(
        self,
        model_or_name_path: str = _DEFAULT_MODEL_V2,
        torch_dtype: torch.dtype = _DEFAULT_TORCH_DTYPE,
        local_files_only: bool = False,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype
        self.model = Dinov2Model.from_pretrained(model_or_name_path, torch_dtype=torch_dtype, local_files_only=local_files_only).to(device)
        self.model.eval()
        self.processor = BitImageProcessor.from_pretrained(model_or_name_path, local_files_only=local_files_only)

    @torch.no_grad()
    def get_image_features(self, image: ImageType | list[ImageType], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model(inputs["pixel_values"].to(self.device, dtype=self.dtype)).last_hidden_state[:, 0, :]
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features


# New metric classes

class FIDScore:
    def __init__(
        self,
        dims: int = 2048,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.dims = dims
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)
        self.model.eval()
    
    def to(self, device: str | torch.device | None = None):
        if device is not None:
            self.device = device
            self.model = self.model.to(device)
    
    @torch.no_grad()
    def get_activations(self, images: list[ImageType]) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]
        
        # Prepare the images
        batch_size = 32
        n_batches = int(np.ceil(len(images) / batch_size))
        n_used_imgs = len(images)
        pred_arr = np.empty((n_used_imgs, self.dims))
        
        for i in range(n_batches):
            start = i * batch_size
            end = min(len(images), (i + 1) * batch_size)
            
            # Convert images to appropriate format
            batch = []
            for j in range(start, end):
                img = images[j]
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                img = img.convert('RGB')
                
                # Resize to InceptionV3 expected input
                img = img.resize((299, 299), Image.BICUBIC)
                img = np.array(img).astype(np.float32)
                img = img.transpose((2, 0, 1))  # to CHW format
                img = img / 255.0  # normalize to [0, 1]
                img = torch.from_numpy(img).unsqueeze(0)
                batch.append(img)
            
            batch = torch.cat(batch, dim=0).to(self.device)
            pred = self.model(batch)[0]
            
            # If model output is not scalar, apply global spatial average pooling
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
                
            pred = pred.squeeze().cpu().numpy()
            pred_arr[start:end] = pred
        
        return pred_arr
    
    def fid_score(self, images1: list[ImageType], images2: list[ImageType]) -> float:
        # Get activations
        act1 = self.get_activations(images1)
        act2 = self.get_activations(images2)
        
        # Calculate mean and covariance
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        
        # Calculate FID
        fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value


class RMSEScore:
    def __init__(self, device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
    
    def rmse_score(self, images1: list[ImageType], images2: list[ImageType]) -> float:
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        
        assert len(images1) == len(images2), "Number of images in both lists should be same"
        
        total_rmse = 0.0
        for img1, img2 in zip(images1, images2):
            # Convert to numpy arrays if they aren't already
            if not isinstance(img1, np.ndarray):
                img1 = np.array(img1)
            if not isinstance(img2, np.ndarray):
                img2 = np.array(img2)
            
            # Resize to match if necessary
            if img1.shape != img2.shape:
                img2 = np.array(Image.fromarray(img2).resize((img1.shape[1], img1.shape[0])))
            
            # Calculate RMSE
            mse = np.mean((img1 - img2) ** 2)
            rmse = np.sqrt(mse)
            total_rmse += rmse
        
        return total_rmse / len(images1)


class SSIMScore:
    def __init__(self, device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
    
    def ssim_score(self, images1: list[ImageType], images2: list[ImageType]) -> float:
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        
        assert len(images1) == len(images2), "Number of images in both lists should be same"
        
        total_ssim = 0.0
        for img1, img2 in zip(images1, images2):
            # Convert to numpy arrays if they aren't already
            if not isinstance(img1, np.ndarray):
                img1 = np.array(img1)
            if not isinstance(img2, np.ndarray):
                img2 = np.array(img2)
            
            # Resize to match if necessary
            if img1.shape != img2.shape:
                img2 = np.array(Image.fromarray(img2).resize((img1.shape[1], img1.shape[0])))
            
            # Convert to grayscale if needed for SSIM
            if len(img1.shape) == 3 and img1.shape[2] > 1:
                img1_gray = np.mean(img1, axis=2).astype(np.uint8)
                img2_gray = np.mean(img2, axis=2).astype(np.uint8)
                ssim_val = ssim(img1_gray, img2_gray)
            else:
                ssim_val = ssim(img1, img2)
            
            total_ssim += ssim_val
        
        return total_ssim / len(images1)


class F1Score:
    def __init__(self, threshold: float = 0.5, device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
        self.threshold = threshold
        self.device = device
    
    def f1_score(self, images1: list[ImageType], images2: list[ImageType]) -> float:
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        
        assert len(images1) == len(images2), "Number of images in both lists should be same"
        
        total_f1 = 0.0
        for img1, img2 in zip(images1, images2):
            # Convert to numpy arrays if they aren't already
            if not isinstance(img1, np.ndarray):
                img1 = np.array(img1)
            if not isinstance(img2, np.ndarray):
                img2 = np.array(img2)
            
            # Resize to match if necessary
            if img1.shape != img2.shape:
                img2 = np.array(Image.fromarray(img2).resize((img1.shape[1], img1.shape[0])))
            
            # Binarize images using threshold
            img1_bin = (img1 > self.threshold * 255).astype(int)
            img2_bin = (img2 > self.threshold * 255).astype(int)
            
            # Flatten arrays
            img1_flat = img1_bin.flatten()
            img2_flat = img2_bin.flatten()
            
            # Calculate F1 score
            f1 = f1_score(img1_flat, img2_flat, average='binary', zero_division=1)
            total_f1 += f1
        
        return total_f1 / len(images1)


class MUSIQScore:
    def __init__(
        self,
        model_name: str = "google/musiq-base",
        local_files_only: bool = LOCAL_FILES_ONLY,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model = AutoModelForImageClassification.from_pretrained(model_name, local_files_only=local_files_only).to(device)
        self.model.eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=local_files_only)
    
    def to(self, device: str | torch.device | None = None):
        if device is not None:
            self.device = device
            self.model = self.model.to(device)
    
    @torch.no_grad()
    def musiq_score(self, images: list[ImageType]) -> float:
        if not isinstance(images, list):
            images = [images]
        
        total_score = 0.0
        for img in images:
            inputs = self.feature_extractor(images=img, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            
            # MUSIQ predicts quality between 0 and 100
            predicted_score = outputs.logits.item()
            total_score += predicted_score
        
        return total_score / len(images)


class MANIQAScore:
    def __init__(
        self,
        model_name: str = "chavinlo/man-iqa",
        local_files_only: bool = LOCAL_FILES_ONLY,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_files_only).to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    
    def to(self, device: str | torch.device | None = None):
        if device is not None:
            self.device = device
            self.model = self.model.to(device)
    
    @torch.no_grad()
    def maniqa_score(self, images: list[ImageType]) -> float:
        if not isinstance(images, list):
            images = [images]
        
        total_score = 0.0
        for img in images:
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            
            # MAN-IQA predicts quality scores
            predicted_score = outputs.logits.item()
            total_score += predicted_score
        
        return total_score / len(images)


def evaluate_metrics(
    image1_dir: str,
    image2_dir: str,
    distributed_state: PartialState | None = None,
    metrics: list[str] = ["rmse", "ssim", "fid", "f1"], #, "maniqa", , "musiq"
    write_to_json: bool = False,
    subject: str = None,
) -> Dict[str, float]:
    """
    Evaluate multiple image metrics between two directories of images.
    
    Args:
        image1_dir: Directory containing the first set of images
        image2_dir: Directory containing the second set of images
        distributed_state: Distributed state for multi-GPU evaluation
        metrics: List of metrics to evaluate
        write_to_json: Whether to write results to JSON files
        subject: Filter images by subject keyword in path
        
    Returns:
        Dictionary of metric scores
    """
    if distributed_state is None:
        distributed_state = PartialState()
    
    device = distributed_state.device
    
    # Initialize metric calculators based on requested metrics
    metric_calculators = {}
    if "dino_v1" in metrics:
        metric_calculators["dino_v1"] = DinoScore(device=device)
    if "dino_v2" in metrics:
        metric_calculators["dino_v2"] = Dinov2Score(device=device)
    if "rmse" in metrics:
        metric_calculators["rmse"] = RMSEScore(device=device)
    if "ssim" in metrics:
        metric_calculators["ssim"] = SSIMScore(device=device)
    if "fid" in metrics:
        metric_calculators["fid"] = FIDScore(device=device)
    if "f1" in metrics:
        metric_calculators["f1"] = F1Score(device=device)
    if "maniqa" in metrics:
        metric_calculators["maniqa"] = MANIQAScore(device=device)
    if "musiq" in metrics:
        metric_calculators["musiq"] = MUSIQScore(device=device)
    
    # Ensure directory paths end with a slash
    if image1_dir[-1] != "/":
        image1_dir = image1_dir + "/"
    if image2_dir[-1] != "/":
        image2_dir = image2_dir + "/"
    
    # Find image files
    image1_files = []
    for _ext in IMAGE_EXT:
        if subject is not None:
            image1_files.extend(megfile.smart_glob(os.path.join(image1_dir, f"**/*{subject}*/**/*.{_ext}")))
        else:
            image1_files.extend(megfile.smart_glob(os.path.join(image1_dir, f"**/*.{_ext}")))
    image1_files = sorted(image1_files)
    
    image2_files = []
    for _ext in IMAGE_EXT:
        if subject is not None:
            image2_files.extend(megfile.smart_glob(os.path.join(image2_dir, f"**/*{subject}*/**/*.{_ext}")))
        else:
            image2_files.extend(megfile.smart_glob(os.path.join(image2_dir, f"**/*.{_ext}")))
    image2_files = sorted(image2_files)
    
    assert len(image1_files) == len(image2_files), f"Number of image1 files {len(image1_files)} != number of image2 files {len(image2_files)}."
    
    # Match file pairs
    params = []
    for image1_file, image2_file in zip(image1_files, image2_files):
        assert (
            image1_file.split(image1_dir)[-1].split(".")[0] == image2_file.split(image2_dir)[-1].split(".")[0]
        ), f"Image1 file {image1_file} and image2 file {image2_file} do not match."
        
        params.append((image1_file, image2_file))
    
    pbar = tqdm(
        total=math.ceil(len(image1_files) / distributed_state.num_processes),
        desc=f"Evaluating Image Metrics",
        disable=not distributed_state.is_local_main_process,
    )
    
    # Initialize result dictionaries
    save_dicts = {metric: {} for metric in metrics}
    metric_scores = {metric: 0.0 for metric in metrics}
    
    # Special handling for FID which needs to process all images together
    if "fid" in metrics:
        all_images1 = []
        all_images2 = []
    
    # Process images in distributed batches
    with distributed_state.split_between_processes(params) as sub_params:
        for _param in sub_params:
            image1_file, image2_file = _param
            save_key = image1_file.split(image1_dir)[-1].split(".")[0].replace("/", "-")
            
            # Load images
            image1 = load_image(image1_file)
            image2 = load_image(image2_file)
            
            # Calculate metrics for each image pair
            for metric in metrics:
                if metric == "dino_v1":
                    score = metric_calculators[metric].dino_score(image1, image2)[0].item()
                    save_dicts[metric][save_key] = score
                    metric_scores[metric] += score
                
                elif metric == "dino_v2":
                    score = metric_calculators[metric].dino_score(image1, image2)[0].item()
                    save_dicts[metric][save_key] = score
                    metric_scores[metric] += score
                
                elif metric == "rmse":
                    score = metric_calculators[metric].rmse_score(image1, image2)
                    save_dicts[metric][save_key] = score
                    metric_scores[metric] += score
                
                elif metric == "ssim":
                    score = metric_calculators[metric].ssim_score(image1, image2)
                    save_dicts[metric][save_key] = score
                    metric_scores[metric] += score
                
                elif metric == "f1":
                    score = metric_calculators[metric].f1_score(image1, image2)
                    save_dicts[metric][save_key] = score
                    metric_scores[metric] += score
                
                elif metric == "maniqa":
                    score = metric_calculators[metric].maniqa_score(image1)
                    save_dicts[metric][save_key] = score
                    metric_scores[metric] += score
                
                elif metric == "musiq":
                    score = metric_calculators[metric].musiq_score(image1)
                    save_dicts[metric][save_key] = score
                    metric_scores[metric] += score
                
                elif metric == "fid":
                    # For FID, collect images first
                    all_images1.append(image1)
                    all_images2.append(image2)
            
            pbar.update(1)
    
    # Calculate FID if required (needs all images at once)
    if "fid" in metrics and len(all_images1) > 0:
        fid_score = metric_calculators["fid"].fid_score(all_images1, all_images2)
        metric_scores["fid"] = fid_score
        for save_key in save_dicts["fid"].keys():
            save_dicts["fid"][save_key] = fid_score  # All get the same global FID score
    
    # Write results to JSON if required
    if write_to_json:
        for metric in metrics:
            with open(f"tmp_{metric}_scores_{distributed_state.process_index}.json", "w") as f:
                json.dump(save_dicts[metric], f, indent=4)
    
    # Gather scores across all processes
    final_scores = {}
    for metric in metrics:
        scores = all_gather(metric_scores[metric])
        if metric == "fid":  # FID is special - just take the first value as they should all be the same
            final_scores[metric] = scores[0]
        else:
            final_scores[metric] = sum(scores) / len(image1_files)
    
    # Write combined JSON files for the main process
    if write_to_json and distributed_state.is_local_main_process:
        for metric in metrics:
            combined_dict = {}
            for i in range(distributed_state.num_processes):
                try:
                    with open(f"tmp_{metric}_scores_{i}.json", "r") as f:
                        _dict = json.load(f)
                        combined_dict.update(_dict)
                    os.remove(f"tmp_{metric}_scores_{i}.json")
                except FileNotFoundError:
                    pass
            
            with open(f"{metric}_scores.json", "w") as f:
                json.dump(combined_dict, f, indent=4)
    
    return final_scores


def multigpu_eval_dino_score(
    image1_dir: str,
    image2_dir: str,
    distributed_state: PartialState | None = None,
    dino_score: DinoScore | Dinov2Score | None = None,
    version: Literal["v1", "v2"] = "v1",
    write_to_json: bool = False,
    subject: str = None,
) -> float:
    if distributed_state is None:
        distributed_state = PartialState()

    if dino_score is None:
        if version == "v1":
            dino_score = DinoScore(device=distributed_state.device)
        elif version == "v2":
            dino_score = Dinov2Score(device=distributed_state.device)
        else:
            raise ValueError(f"Invalid version {version}")

    if image1_dir[-1] != "/":
        image1_dir = image1_dir + "/"

    if image2_dir[-1] != "/":
        image2_dir = image2_dir + "/"

    image1_files = []
    for _ext in IMAGE_EXT:
        if subject is not None:
            image1_files.extend(megfile.smart_glob(os.path.join(image1_dir, f"**/*{subject}*/**/*.{_ext}")))
        else:
            image1_files.extend(megfile.smart_glob(os.path.join(image1_dir, f"**/*.{_ext}")))
    image1_files = sorted(image1_files)
    
    image2_files = []
    for _ext in IMAGE_EXT:
        if subject is not None:
            image2_files.extend(megfile.smart_glob(os.path.join(image2_dir, f"**/*{subject}*/**/*.{_ext}")))
        else:
            image2_files.extend(megfile.smart_glob(os.path.join(image2_dir, f"**/*.{_ext}")))

    image2_files = sorted(image2_files)
    
    assert len(image1_files) == len(image2_files), f"Number of image1 files {len(image1_files)} != number of image2 files {len(image2_files)}."

    params = []
    for image1_file, image2_file in zip(image1_files, image2_files):
        assert (
            image1_file.split(image1_dir)[-1].split(".")[0] == image2_file.split(image2_dir)[-1].split(".")[0]
        ), f"Image1 file {image1_file} and image2 file {image2_file} do not match."

        params.append((image1_file, image2_file))

    pbar = tqdm(
        total=math.ceil(len(image1_files) / distributed_state.num_processes),
        desc=f"Evaluating Dino{version} Score",
        disable=not distributed_state.is_local_main_process,
    )

    save_dict = {}
    with distributed_state.split_between_processes(params) as sub_params:
        score = 0
        for _param in sub_params:
            image1_file, image2_file = _param
            save_key = image1_file.split(image1_dir)[-1].split(".")[0].replace("/", "-")
            image1, image2 = load_image(image1_file), load_image(image2_file)
            _score = dino_score.dino_score(image1, image2)[0]
            save_dict[save_key] = _score.item()
            score += _score

            pbar.update(1)

    if write_to_json:
        with open(f"tmp_dino_scores_{distributed_state.process_index}.json", "w") as f:
            json.dump(save_dict, f, indent=4)
    scores = all_gather(score)
    if write_to_json:
        if distributed_state.is_local_main_process:
            save_dict = {}
            for i in range(distributed_state.num_processes):
                with open(f"tmp_dino_scores_{i}.json", "r") as f:
                    _dict = json.load(f)
                    save_dict.update(_dict)
                os.remove(f"tmp_dino_scores_{i}.json")
            with open(f"dino_scores_{version}.json", "w") as f:
                json.dump(save_dict, f, indent=4)
    return (sum(scores) / len(image1_files)).item()


def eval_all(
    dir1: str, 
    dir2: str, 
    metrics: list[str] = ["dino_v1", "dino_v2", "rmse", "ssim", "fid", "f1", "maniqa", "musiq"],
    subject: str = None
):
    """
    Run all specified metrics evaluations and print results
    """
    results = evaluate_metrics(
        dir1, 
        dir2,
        metrics=metrics,
        write_to_json=True,
        subject=subject
    )
    
    # Print results with appropriate arrows
    logger.info("Evaluation Results:")
    for metric, score in results.items():
        if metric in ["dino_v1", "dino_v2", "ssim", "f1", "maniqa", "musiq"]:
            # Higher is better
            logger.info(f"{metric.upper()}: {score:.4f} ↑")
        else:
            # Lower is better
            logger.info(f"{metric.upper()}: {score:.4f} ↓")
    
    return results

if __name__ == "__main__":
    fire.Fire(eval_all)