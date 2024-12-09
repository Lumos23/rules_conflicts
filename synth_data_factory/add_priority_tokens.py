import json
from constants import rules

def process_rules_in_prompt(prompt, source_type, rules, rid):
    """
    Process the rules in the prompt to wrap rule1 of the specified source_type with priority tokens
    """
    # Get rule1 for the source type
    if source_type not in rules:
        return prompt
    
    rule = rules[source_type][rid-1]
    # Split the prompt into lines
    lines = prompt.split('\n')
    
    # Find and wrap the matching rule
    for i, line in enumerate(lines):
        if line.strip() == rule:
            lines[i] = f"<|priority_start|> {line} <|priority_end|>"
    
    return '\n'.join(lines)

def process_json_file(input_file, output_file):
    """Process the JSON file to add priority tokens around rule1 based on source type"""
    # Load rules
    
    # Read input JSON
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    fname = input_file.split('/')[-1].split('.')[0]
    print(fname)
    if 'rule1' in fname:
        rid = 1
    elif 'rule2' in fname:
        rid = 2
    
    # Process each entry
    for entry in data:
        source_type = entry.get('source')
        if source_type:
            entry['prompt'] = process_rules_in_prompt(entry['prompt'], source_type, rules, rid)
        else:
            raise ValueError(f"Source type not found for entry: {entry}")
        
    # Write output JSON
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

# Example usage
if __name__ == "__main__":
    input_file = "/scratch/gpfs/lh2046/rules_conflicts/data/training_data/raw_categorized/rule2_only.json"
    fname = input_file.split('/')[-1].split('.')[0]
    output_file = f"/scratch/gpfs/lh2046/rules_conflicts/data/training_data/with_priority/{fname}_with_priority.json"
    process_json_file(input_file, output_file)