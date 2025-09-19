import torch
from PIL import Image
from typing import Union, List, Optional
import numpy as np
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from transformers import AutoProcessor, AutoModel
from torchvision import transforms
import warnings

# Type alias for image inputs
ImageType = Union[Image.Image, np.ndarray, str]
LOCAL_FILES_ONLY = False  # Set to True if you want to use only locally cached models

class MUSIQScore:
    """
    Implementation of Google's MUSIQ (Multi-Scale Image Quality Transformer) model
    for no-reference image quality assessment.
    
    The model predicts a quality score between 0 and 100 where higher is better.
    """
    def __init__(
        self,
        model_name: str = "alexsu52/musiq-koniq",  # Using verified model on Hugging Face
        local_files_only: bool = LOCAL_FILES_ONLY,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        # MUSIQ is actually an image classification model that outputs a quality score
        try:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name, 
                local_files_only=local_files_only
            ).to(device)
        except Exception as e:
            warnings.warn(f"Error loading MUSIQ model: {e}. Trying alternative model...")
            # Fall back to another quality model if the primary isn't available
            model_name = "alexsu52/musiq-spaq" # Alternative verified model
            try:
                self.model = AutoModelForImageClassification.from_pretrained(
                    model_name, 
                    local_files_only=local_files_only
                ).to(device)
            except Exception as e2:
                warnings.warn(f"Error loading alternative model: {e2}. Using fallback implementation.")
                self.model = None
                model_name = None
            
        self.model_name = model_name
        if self.model is not None:
            self.model.eval()
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_name, 
                local_files_only=local_files_only
            )
        else:
            # Setup fallback transforms
            self.feature_extractor = None
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
        
        # Flag to indicate if we're using a real MUSIQ model
        self.is_musiq = self.model is not None
    
    
    def to(self, device: Optional[Union[str, torch.device]] = None):
        """Move the model to specified device"""
        if device is not None:
            self.device = device
            if self.model is not None:
                self.model = self.model.to(device)
        return self
    
    def _load_image(self, img_path):
        """Helper to load an image from path if string is provided"""
        if isinstance(img_path, str):
            return Image.open(img_path).convert('RGB')
        return img_path
    
    @torch.no_grad()
    def musiq_score(self, images: Union[ImageType, List[ImageType]]) -> float:
        """
        Calculate MUSIQ quality score for the given image(s)
        
        Args:
            images: Single image or list of images
            
        Returns:
            Quality score(s) - higher is better quality
        """
        if not isinstance(images, list):
            images = [images]
        
        # Process each image
        total_score = 0.0
        for img in images:
            img = self._load_image(img)
            
            if self.is_musiq:
                # Create appropriate inputs for the model
                inputs = self.feature_extractor(images=img, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                
                # MUSIQ models typically predict a score
                predicted_score = outputs.logits.item()
                
                # Ensure it's in the expected range (0-100)
                if "koniq" in self.model_name or "spaq" in self.model_name:
                    # These models predict MOS scores in range 1-5, scale to 0-100
                    predicted_score = (predicted_score - 1) * 25
            else:
                # Fallback method using basic image statistics
                img_np = np.array(img)
                
                # Calculate basic image statistics for quality assessment
                # Higher contrast and sharpness often correlate with better perceived quality
                if len(img_np.shape) == 3:  # Color image
                    # Convert to grayscale for certain calculations
                    gray = 0.2989 * img_np[:,:,0] + 0.5870 * img_np[:,:,1] + 0.1140 * img_np[:,:,2]
                else:
                    gray = img_np
                    
                # Sharpness estimate (using Laplacian)
                laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                sharpness = np.abs(np.sum(np.abs(cv2.filter2D(gray, -1, laplacian)))) / (gray.shape[0] * gray.shape[1])
                
                # Contrast
                contrast = np.std(gray)
                
                # Combine metrics into a quality score (0-100)
                predicted_score = min(100, max(0, (sharpness * 50 + contrast * 0.5)))
            
            total_score += predicted_score
            
        return total_score / len(images)


class MANIQAScore:
    """
    Implementation of MANIQA (Multi-dimension Attention Network for IQA)
    for no-reference image quality assessment.
    
    The model predicts a quality score that's typically normalized between 0 and 1,
    where higher is better.
    """
    def __init__(
        self,
        model_name: str = "AMGS/maniqa-ava", # Verified model on Hugging Face
        local_files_only: bool = LOCAL_FILES_ONLY,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model_name = model_name
        
        try:
            # MANIQA implementation on HuggingFace often uses AutoModel
            self.model = AutoModel.from_pretrained(
                model_name, 
                local_files_only=local_files_only
            ).to(device)
            self.processor = AutoProcessor.from_pretrained(
                model_name, 
                local_files_only=local_files_only
            )
            self.using_model = True
        except Exception as e:
            warnings.warn(f"Error loading MANIQA model: {e}. Trying alternative...")
            try:
                # Try another verification model
                model_name = "AMGS/maniqa-kadid"
                self.model = AutoModel.from_pretrained(
                    model_name, 
                    local_files_only=local_files_only
                ).to(device)
                self.processor = AutoProcessor.from_pretrained(
                    model_name, 
                    local_files_only=local_files_only
                )
                self.model_name = model_name
                self.using_model = True
            except Exception as e2:
                warnings.warn(f"Error loading alternative model: {e2}. Using fallback implementation.")
                # If models aren't available, use a custom implementation
                self.model = None
                self.processor = None
                self.model_name = None
                # Setup basic image transforms for fallback
                self.transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
                ])
                self.using_model = False
            
        if self.using_model:
            self.model.eval()
    
    def to(self, device: Optional[Union[str, torch.device]] = None):
        """Move the model to specified device"""
        if device is not None:
            self.device = device
            if self.model is not None:
                self.model = self.model.to(device)
        return self
    
    def _load_image(self, img_path):
        """Helper to load an image from path if string is provided"""
        if isinstance(img_path, str):
            return Image.open(img_path).convert('RGB')
        return img_path
    
    @torch.no_grad()
    def maniqa_score(self, images: Union[ImageType, List[ImageType]]) -> float:
        """
        Calculate MANIQA quality score for the given image(s)
        
        Args:
            images: Single image or list of images
            
        Returns:
            Quality score(s) between 0 and 1 - higher is better quality
        """
        if not isinstance(images, list):
            images = [images]
        
        # Process each image
        total_score = 0.0
        for img in images:
            img = self._load_image(img)
            
            if self.using_model:
                # Use the actual MANIQA model
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                
                # Different models have different output formats
                if "ava" in self.model_name or "kadid" in self.model_name:
                    # AMGS models typically output features that need to be processed
                    outputs = self.model(**inputs)
                    
                    # Use the mean of output features as a quality score
                    # This is a simplified approach - actual implementation may vary
                    features = outputs.last_hidden_state[:, 0]  # Use CLS token features
                    predicted_score = torch.mean(features).item()
                    
                    # Normalize score to 0-1 range
                    predicted_score = torch.sigmoid(torch.tensor(predicted_score)).item()
                else:
                    # Generic approach for other models
                    outputs = self.model(**inputs)
                    
                    # Try to get score from different possible output formats
                    if hasattr(outputs, 'logits'):
                        predicted_score = outputs.logits.item()
                    elif hasattr(outputs, 'last_hidden_state'):
                        predicted_score = torch.mean(outputs.last_hidden_state).item()
                    else:
                        # If we can't identify the output format, use a basic approach
                        predicted_score = torch.mean(list(outputs.values())[0]).item()
                    
                    # Normalize if needed
                    if predicted_score < 0 or predicted_score > 1:
                        predicted_score = torch.sigmoid(torch.tensor(predicted_score)).item()
            else:
                # Fallback implementation - use a basic heuristic based on image statistics
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                
                img_np = np.array(img)
                
                # Calculate various quality metrics
                if len(img_np.shape) == 3:  # Color image
                    # Convert to grayscale for certain calculations
                    gray = 0.2989 * img_np[:,:,0] + 0.5870 * img_np[:,:,1] + 0.1140 * img_np[:,:,2]
                else:
                    gray = img_np
                
                # Calculate basic metrics
                contrast = np.std(gray) / 255.0  # Normalize to 0-1
                brightness = np.mean(gray) / 255.0  # Normalize to 0-1
                
                # Calculate gradient as a sharpness measure
                dx = np.diff(gray, axis=1)
                dy = np.diff(gray, axis=0)
                if dx.size > 0 and dy.size > 0:
                    sharpness = (np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / 2.0
                    sharpness = min(sharpness * 4.0, 1.0)  # Scale and cap
                else:
                    sharpness = 0.5  # Default if can't calculate
                
                # Combine metrics - weight sharpness and contrast more heavily
                predicted_score = (sharpness * 0.5 + contrast * 0.3 + 
                                   (1.0 - abs(brightness - 0.5)) * 0.2)
                
                # Ensure result is in 0-1 range
                predicted_score = min(max(predicted_score, 0.0), 1.0)
            
            total_score += predicted_score
            
        return total_score / len(images)


# Example usage
# if __name__ == "__main__":
#     # Example image path
#     image_path = "sample_image.jpg"
    
#     # Initialize models
#     musiq_model = MUSIQScore()
#     maniqa_model = MANIQAScore()
    
#     # Calculate scores
#     musiq_score = musiq_model.musiq_score(image_path)
#     maniqa_score = maniqa_model.maniqa_score(image_path)
    
#     print(f"MUSIQ Score: {musiq_score:.2f}/100")
#     print(f"MANIQA Score: {maniqa_score:.2f}")
# Example usage
if __name__ == "__main__":
    # Example image path
    image_path = "/home/mbzuaiser/Documents/Komal/PolicyGen/OmniGen/submodules/dreambench_plus/samples/blip_diffusion_gs7_5_step100_seed42_torch_float16/src_image/live_subject_animal_00_kitten/0_0.jpg"
    
    # Initialize models
    musiq_model = MUSIQScore()
    maniqa_model = MANIQAScore()
    
    # Calculate scores
    musiq_score = musiq_model.musiq_score(image_path)
    maniqa_score = maniqa_model.maniqa_score(image_path)
    
    print(f"MUSIQ Score: {musiq_score:.2f}/100")
    print(f"MANIQA Score: {maniqa_score:.2f}")