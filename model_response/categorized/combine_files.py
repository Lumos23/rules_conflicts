import json
import glob
import os
import argparse
from pathlib import Path

def read_existing_file(filename):
    """
    Attempts to read an existing JSON file and returns its content.
    Returns None if file doesn't exist or isn't valid JSON.
    """
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
    except Exception:
        return None
    return None

def combine_json_files(pattern, clear_existing=False):
    """
    Combines all JSON files matching the given pattern into a single output file.
    
    Args:
        pattern: File pattern to match
        clear_existing: If True, overwrites existing files. If False, merges with existing content.
    """
    # Modified pattern matching to look for files ending with the pattern
    files = glob.glob(f"*{pattern}")
    
    # Print found files for debugging
    print(f"\nFound files matching pattern '{pattern}':")
    for file in files:
        print(f"  - {file}")
    
    combined_data = []
    
    # Check if output file exists and handle existing content
    output_file = pattern
    existing_data = read_existing_file(output_file)
    
    if existing_data and not clear_existing:
        print(f"Warning: {output_file} already exists and contains data!")
        user_input = input(f"Do you want to merge with existing content in {output_file}? (y/n): ").lower()
        if user_input != 'y':
            print(f"Skipping {pattern}")
            return
        
        # Add existing data if it's not being cleared
        if isinstance(existing_data, list):
            combined_data.extend(existing_data)
        else:
            combined_data.append(existing_data)
    
    # Read and combine data from all matching files
    for file in files:
        # Skip the output file if it matches the pattern
        if file == output_file:
            continue
            
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                # Handle both single objects and lists
                if isinstance(data, list):
                    combined_data.extend(data)
                else:
                    combined_data.append(data)
            print(f"Successfully read: {file}")
        except json.JSONDecodeError as e:
            print(f"Error reading {file}: {e}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Write combined data to output file
    if combined_data:
        try:
            with open(output_file, 'w') as f:
                json.dump(combined_data, f, indent=2)
            print(f"Successfully combined {len(files)} files into {output_file}")
        except Exception as e:
            print(f"Error writing to {output_file}: {e}")
    else:
        print(f"No data found for pattern: {pattern}")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Combine JSON files based on filename patterns')
    parser.add_argument('--clear', action='store_true', 
                      help='Clear existing content in output files before writing')
    parser.add_argument('--patterns', nargs='+', 
                      default=["rule1_only.json", "rule2_only.json", "neither.json", "both.json"],
                      help='Patterns to match (default: rule1_only.json rule2_only.json neither.json both.json)')
    
    args = parser.parse_args()
    
    # Process each pattern
    for pattern in args.patterns:
        print(f"\nProcessing files matching: {pattern}")
        combine_json_files(pattern, args.clear)

if __name__ == "__main__":
    main()

    # use examples 

# Default run (interactive prompts if files exist)
# python combine_json.py

# # Clear existing files without prompting
# python combine_json.py --clear

# # Specify custom patterns
# python combine_json.py --patterns rule1_only.json rule2_only.json

# # Both clear and custom patterns
# python combine_json.py --clear --patterns rule1_only.json rule2_only.json