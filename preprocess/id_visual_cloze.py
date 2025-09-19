import os

def last_three(path_input: str, path_output: str) -> str:
    # Extract the last two components of input and output paths
    input_last = os.path.basename(path_input).split('.')[0]  # '0_ref'
    output_last = os.path.basename(path_output).split('.')[0]  # '0_target'

    # Extract the directory names for input and output
    input_dir = os.path.dirname(path_input).split('/')[-1]  # 'ref'
    output_dir = os.path.dirname(path_output).split('/')[-1]  # 'target'

    # Combine them into the required structure
    new_id = f"{input_dir}/{input_last}/{output_dir}/{output_last}"
    
    print(new_id)
    return new_id

# Example usage
path_input = "/media/mbzuaiser/SSD1/Komal/Graph200K/test/ref/0_ref.jpg"
path_output = "/media/mbzuaiser/SSD1/Komal/Graph200K/test/target/0_target.jpg"

last_three(path_input, path_output)