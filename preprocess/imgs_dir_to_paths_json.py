import os
import json

# Define the root directory where the images are located

import os
import json

def create_jsonl_for_images(root_dir, output_file):
    """
    Create a JSONL file containing paths to images in the 'images' subdirectories.
    
    Args:
        root_dir (str): Root directory to search for images
        output_file (str): Path to the output JSONL file
    """
    with open(output_file, 'w') as f:
        # Walk through the directory structure
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Check if this is an 'images' directory
            if os.path.basename(dirpath) == 'images':
                # Get the relative path from the root directory
                rel_path = os.path.relpath(dirpath, root_dir)
                parent_dir = os.path.dirname(rel_path)
                
                # Process each file in the directory
                for filename in filenames:
                    # Only include image files (you can add more extensions if needed)
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        # Create the relative path for the output_image field
                        image_path = os.path.join(parent_dir, 'images', filename)
                        
                        # Create the JSON object
                        json_obj = {
                            "task_type": "text_to_image",
                            "output_image": image_path
                        }
                        
                        # Write the JSON object to the file
                        f.write(json.dumps(json_obj) + '\n')
    
    print(f"JSONL file created at: {output_file}")

if __name__ == "__main__":
    # Set the root directory and output file path
    root_dir = "/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/Objects/"
    output_file = "/nvme-data/Komal/documents/omni_datasets/InsDet-FULL/InsDet-FULL-full-objects.jsonl"

    create_jsonl_for_images(root_dir, output_file)