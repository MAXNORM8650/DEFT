import json
import os
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Set the root directory where images are stored
image_root = "/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/"

# Check for CUDA availability
if torch.cuda.is_available():
    device = "cuda"
    print(f"CUDA is available. Found {torch.cuda.device_count()} GPUs.")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    device = "cpu"
print(f"Using device: {device}")

# Load Qwen2.5-VL-32B model and processor
model_name = "Qwen/Qwen2.5-VL-32B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)

# Load the model with memory optimization techniques
# Set device_map to "auto" to let transformers handle the distribution
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Using bfloat16 precision to reduce memory usage
    device_map="auto",           # Let the library handle device mapping
    # max_memory={0: "48GiB", 1: "48GiB", 2: "48GiB", 3: "1GiB"},  # Allocate memory for each GPU
    # low_cpu_mem_usage=True       # Reduce CPU memory usage during loading
)

# Get the primary device where most of the model is loaded
# primary_device = next(model.parameters()).device
# print(f"Model's primary device: {primary_device}")

# Function to generate a caption for an image using Qwen2.5-VL-32B
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Create messages for image captioning
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "Describe this image in detail. Keep it concise."},
                ],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Process inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move all inputs to the primary device of the model
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(model.device)
        
        # Generate the caption with appropriate memory settings
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=128,
                use_cache=True
            )
            
            # Ensure generated_ids and input_ids are on the same device for subtraction
            if inputs.input_ids.device != generated_ids.device:
                generated_ids = generated_ids.to(inputs.input_ids.device)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        # Clear CUDA cache after each processing to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return output_text.strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return f"Unable to generate caption for this image: {str(e)}"

# Function to update captions in the existing JSON file
def update_captions_in_json(input_json_file, output_json_file):
    # Read the existing JSONL file
    with open(input_json_file, 'r') as f:
        lines = f.readlines()
    
    captions_data = [json.loads(line) for line in lines]
    total_images = len(captions_data)
    
    print(f"Processing {total_images} images...")
    
    # Process images in batches to better manage memory
    batch_size = 10  # Save results every 10 images
    for i, entry in enumerate(captions_data):
        if i % batch_size == 0:
            print(f"Progress: {i}/{total_images}")
            # Save intermediate results to prevent losing all progress on failure
            if i > 0:
                with open(output_json_file + f".part_{i}", 'w') as f:
                    for j in range(i):
                        f.write(json.dumps(captions_data[j]) + "\n")
            
        # Generate caption for the image
        image_path = entry["output_image"]
        full_image_path = os.path.join(image_root, image_path)  # Full path to the image
        
        # Check if file exists
        if not os.path.exists(full_image_path):
            print(f"Warning: Image not found at {full_image_path}")
            entry["instruction"] = "Image not found"
            continue
            
        # Generate caption and update entry
        caption = generate_caption(full_image_path)
        entry["instruction"] = f"a photo of {caption}"
    
    # Write updated data to the final JSON file (JSONL format: one JSON object per line)
    with open(output_json_file, 'a') as f:
        for entry in captions_data:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Completed: {total_images} images processed.")

# Specify the input JSONL file path (the file containing image data)
input_json_file = "/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/images_objects_blip2.jsonl"

# Specify the output JSON file path where updated captions will be saved
output_json_file = "/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/images_objectss_qwen.json"

# Call the function to update captions and save them in the new JSON file
update_captions_in_json(input_json_file, output_json_file)

print(f"Updated captions saved to {output_json_file}")