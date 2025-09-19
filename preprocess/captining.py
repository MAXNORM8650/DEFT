import os
import datasets
from PIL import Image
import json
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_dataset_components(dataset_path, output_dir, num_examples=1):
    """
    Load examples from a dataset and save each component to the specified directory.
    
    Args:
        dataset_path (str): Path to the dataset
        output_dir (str): Directory to save components
        num_examples (int): Number of examples to process
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from: {dataset_path}")
        dataset = datasets.load_dataset(
            dataset_path,
            split="train",
            streaming=True
        )
        
        # Process specified number of examples
        example_count = 0
        for example in dataset:
            example_dir = os.path.join(output_dir, f"example_{example_count}")
            os.makedirs(example_dir, exist_ok=True)
            
            logger.info(f"Processing example {example_count}")
            save_example_components(example, example_dir)
            
            example_count += 1
            # if example_count >= num_examples:
            #     break
                
        logger.info(f"Successfully processed {example_count} examples")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")

def save_example_components(example, save_dir):
    """
    Save all components of a dataset example to disk.
    
    Args:
        example (dict): The dataset example
        save_dir (str): Directory to save components
    """
    for key, value in example.items():
        try:
            file_path = os.path.join(save_dir, key)
            
            # Handle PIL Images
            if isinstance(value, Image.Image) or str(type(value)).find('PIL') >= 0:
                value.save(f"{file_path}.jpg")
                logger.info(f"Saved {key}.jpg")
                
            # Handle numpy arrays
            elif isinstance(value, np.ndarray):
                if value.dtype == np.uint8 and (value.ndim == 2 or (value.ndim == 3 and value.shape[2] in [1, 3, 4])):
                    # Likely an image array
                    Image.fromarray(value).save(f"{file_path}.jpg")
                    logger.info(f"Saved {key}.jpg (from numpy array)")
                else:
                    # Other numpy arrays
                    np.save(f"{file_path}.npy", value)
                    logger.info(f"Saved {key}.npy")
                    
            # Handle dictionaries and lists
            elif isinstance(value, (dict, list)):
                with open(f"{file_path}.json", 'w') as f:
                    json.dump(value, f, indent=2)
                logger.info(f"Saved {key}.json")
                
            # Handle strings
            elif isinstance(value, str):
                with open(f"{file_path}.txt", 'w') as f:
                    f.write(value)
                logger.info(f"Saved {key}.txt")
                
            # Handle numeric types
            elif isinstance(value, (int, float, bool)):
                with open(f"{file_path}.txt", 'w') as f:
                    f.write(str(value))
                logger.info(f"Saved {key}.txt")
                
            # Handle other types
            else:
                logger.warning(f"Unsupported type for {key}: {type(value)}")
                with open(f"{file_path}.txt", 'w') as f:
                    f.write(str(value))
                logger.info(f"Saved {key}.txt (as string representation)")
                
        except Exception as e:
            logger.error(f"Error saving {key}: {e}")

if __name__ == "__main__":
    # Configuration
    dataset_path = "/media/mbzuaiser/SSD1/Komal/Graph200K"
    output_dir = "/media/mbzuaiser/SSD1/Komal/Graph200K/dataset"

    # Save dataset components
    save_dataset_components(dataset_path, output_dir, num_examples=1)