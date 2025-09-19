import torch
import json
import random
from typing import List, Dict, Any
from transformers import pipeline

class LlamaPromptVariationGenerator:
    """
    Generate prompt variations using LLaMa 3.2 while preserving the key objects
    """
    
    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct"):
        """
        Initialize the generator with a LLaMa model
        """
        self.model_id = model_id
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        self.system_message = """
        You are an assistant for image personalization in diverse contexts. Given a set of descriptions, you will generate multiple prompts that keep the main object but modify the context, style, or scenario. Your goal is to create varied and creative prompts based on the same object, ensuring each instruction is unique while maintaining the object. Each prompt should reflect a different scenario, environment, or artistic style.

        Return the output in the following JSON format:
        {
            "1": "Generated instruction 1",
            "2": "Generated instruction 2",
            "3": "Generated instruction 3",
            ...
        }

        Ensure the following:
        - Preserve the main object but change its context.
        - Provide clear and diverse instructions.
        - Do not include any explanations, special characters, or additional information in your output.
        - Instructions should be short, concise, and different from one another.

        The object description must remain the same, but the surroundings, actions, or scenarios should vary significantly.
        """
    
    def generate_variations(self, instruction: str, num_variations: int = 5, category_types=None) -> Dict[str, str]:
        """
        Generate variations of the given instruction using LLaMa 3.2
        """
        # Create a more specific system message based on the number of variations
        specific_system_message = self.system_message + f"\nGenerate exactly these {category_types} different variations."
        
        messages = [
            {"role": "system", "content": specific_system_message},
            {"role": "user", "content": instruction},
        ]
        
        outputs = self.pipe(
            messages,
            max_new_tokens=512,
            temperature=0.7,  # Add some randomness for creativity
        )
        
        # Extract the generated text
        assistant_content = outputs[0]['generated_text'][-1]['content']
        # Parse the JSON content
        try:
            if assistant_content:
                variations = json.loads(assistant_content)
                return variations
            else:
                # Fallback if we can't extract proper JSON
                return {str(i+1): f"Variation {i+1} of {instruction}" for i in range(num_variations)}
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {str(i+1): f"Variation {i+1} of {instruction}" for i in range(num_variations)}
    
    def create_experiments_from_variations(
        self, 
        instruction: str, 
        output_image: str,
        num_variations: int = 5,
        category_types: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create experiment entries from variations of the instruction
        
        Args:
            instruction: The original instruction
            output_image: The path to the output image (which will be used as input reference)
            num_variations: Number of variations to generate
            category_types: List of category types to use (defaults to "consistent_object_different_scenes")
            
        Returns:
            Dictionary with experiments list
        """
        # Default category if none provided
        if not category_types:
            category_types = ["consistent_object_different_scenes"]
        
        # Generate variations
        variations = self.generate_variations(instruction, num_variations, category_types)
        
        # Create experiments
        experiments = []
        
        for idx, prompt in variations.items():
            # Randomly select a category type
            category = random.choice(category_types)
            
            # Create an experiment entry
            experiment = {
                "name": f"{category}_{idx}",
                "prompt": prompt,
                "input_images": [output_image],  # Use the output_image as reference
                "params": {
                    "guidance_scale": round(random.uniform(2.0, 5.0), 1),
                    "seed": random.randint(100, 999)
                }
            }
            
            experiments.append(experiment)
        
        return {"experiments": experiments}

def process_data_batch(
    data_list: List[Dict[str, Any]],
    output_file: str = "evaluation_experiments.json",
    variations_per_item: int = 3,
    category_types: List[str] = None
) -> None:
    """
    Process a batch of data items to create a comprehensive evaluation framework
    
    Args:
        data_list: List of data items with task_type, output_image, and instruction
        output_file: Path to save the output JSON
        variations_per_item: Number of variations to generate per item
        category_types: List of category types to use
    """
    # Default categories if none provided

    if not category_types:
        category_types = [
            "consistent_object_different_scenes",
            "style_transfer",
            "scene_transformation",
            "contextual_modification",
            "artistic_interpretation",
            "object_transformation",
        ]
    
    # Initialize the generator
    generator = LlamaPromptVariationGenerator()
    
    # Collect all experiments
    all_experiments = []
    
    for item in data_list:
        # Skip items that don't have the required fields
        if "instruction" not in item or "output_image" not in item:
            continue
        
        # Generate experiments for this item
        result = generator.create_experiments_from_variations(
            instruction=item["instruction"],
            output_image=item["output_image"],
            num_variations=variations_per_item,
            category_types=category_types
        )
        
        # Add the experiments to the collection
        all_experiments.extend(result["experiments"])
    
    # Save the results
    with open(output_file, "w") as f:
        json.dump({"experiments": all_experiments}, f, indent=4)
    
    print(f"Generated {len(all_experiments)} experiments saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Load the data from a JSON file if needed
    with open("/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/images_objectss_qwen.jsonl", "r") as f:
        # sample_data = json.load(f)
        sample_data = [json.loads(line) for line in f.readlines()]
        sample_data = [entry for entry in sample_data if entry["output_image"].endswith("001.jpg")]

    # breakpoint()
    
    # Process the data
    process_data_batch(
        data_list=sample_data,
        output_file="llama_generated_experiments.json",
        variations_per_item=3
    )
    
    # For a single example
    # generator = LlamaPromptVariationGenerator()
    
    # # Generate variations for one item
    # instruction = "a photo of a mouse is sitting on top of a qr code"
    # output_image = "Objects/075_mouse_thinkpad/images/022.jpg"
    
    # variations = generator.generate_variations(instruction, num_variations=5)
    # print("Generated variations:", json.dumps(variations, indent=4))
    
    # experiments = generator.create_experiments_from_variations(
    #     instruction=instruction,
    #     output_image=output_image,
    #     num_variations=5
    # )
    # print("Generated experiments:", json.dumps(experiments, indent=4))