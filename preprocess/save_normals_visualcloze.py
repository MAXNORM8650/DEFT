import json

def filter_jsonl_by_task_type(input_file, output_file, target_task_type="normal"):
    """
    Filter JSONL file to keep only entries with specified task_type
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output filtered JSONL file
        target_task_type (str): Task type to filter for (default: "normal")
    """
    filtered_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            total_count += 1
            try:
                # Parse JSON object from line
                data = json.loads(line.strip())
                
                # Check if task_type matches target
                if data.get("task_type") == target_task_type:
                    # Write filtered entry to output file
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    filtered_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {total_count}: {e}")
                continue
    
    print(f"Filtered {filtered_count} entries out of {total_count} total entries")
    print(f"Filtered data saved to: {output_file}")

# Usage
if __name__ == "__main__":
    input_path = "/media/mbzuaiser/SSD1/Komal/Graph200K/VisualCloze/trainomnijson.jsonl"
    output_path = "/media/mbzuaiser/SSD1/Komal/Graph200K/VisualCloze/trainomnijson_depth_only.jsonl"
    
    filter_jsonl_by_task_type(input_path, output_path, "depth")