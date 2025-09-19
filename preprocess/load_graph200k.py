import os
import datasets
from PIL import Image
import json


# Load one example
dataset = datasets.load_dataset(
    "/media/mbzuaiser/SSD1/Komal/Graph200K", 
    split="train",
    streaming=True
)

for example in dataset:
    break

import os
saving_dir = "/media/mbzuaiser/SSD1/Komal/Graph200K/dataset/"
os.makedirs(saving_dir, exist_ok=True)
# breakpoint()

for key in example.keys():
    try:
        if isinstance(example[key], Image.Image) or str(type(example[key])).find('PIL') >= 0:
            example[key].save(f"{saving_dir}/{key}.jpg")
            print(f"Saved {key}.jpg")
        elif isinstance(example[key], (dict, list)):
            import json
            with open(f"{saving_dir}/{key}.json", 'w') as f:
                json.dump(example[key], f)
            print(f"Saved {key}.json")
        else:
            with open(f"{saving_dir}/{key}.txt", 'w') as f:
                f.write(str(example[key]))
            print(f"Saved {key}.txt")
    except Exception as e:
        print(f"Error saving {key}: {e}")