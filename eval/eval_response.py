'''
Code to evaluate the responses to check rule compliance, and save prompt/ response pairs to files based on rule compliance.
Categories: rule1_only, rule2_only, both, neither
'''
import json
import csv
from pathlib import Path
import sys
sys.path.append("/scratch/gpfs/lh2046/rules_conflicts")
from synth_data_factory.constants import rules
from synth_data_factory.checkers import *
from copy import deepcopy

def evaluate_response(response: str, source: str) -> tuple[bool, bool]:
    """Evaluate a single response based on source type"""
    checker1 = globals()[f"check_{source}_rule1"]
    checker2 = globals()[f"check_{source}_rule2"]
    
    return checker1(response), checker2(response)

def process_file(filepath: Path, output_dir: Path) -> dict:
    """Process a single json file and return statistics"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Initialize counters and category entries
    stats = {
        'overall': {'rule1_only': 0, 'rule2_only': 0, 'both': 0, 'neither': 0},
    }
    
    categorized_entries = {
        'rule1_only': [],
        'rule2_only': [],
        'both': [],
        'neither': []
    }
    
    # Process each entry
    for entry in data:
        source = entry['source']
        
        # Initialize source-specific counters if not exists
        if source not in stats:
            stats[source] = {'rule1_only': 0, 'rule2_only': 0, 'both': 0, 'neither': 0}
        
        # Process each response separately
        for response in entry['responses']:
            # Create a new entry for this response
            new_entry = deepcopy(entry)
            new_entry['responses'] = response
            
            rule1, rule2 = evaluate_response(response, source)
            
            # Categorize the result
            category = 'both' if rule1 and rule2 else \
                      'rule1_only' if rule1 else \
                      'rule2_only' if rule2 else \
                      'neither'
            
            # Update counters
            stats['overall'][category] += 1
            stats[source][category] += 1
            
            # Add entry to appropriate category
            categorized_entries[category].append(new_entry)
    
    # Write categorized entries to separate files
    for category, entries in categorized_entries.items():
        if entries:  # Only write if there are entries
            category_file = output_dir / f"{filepath.stem}_{category}.json"
            with open(category_file, 'w') as f:
                json.dump(entries, f, indent=4)
    
    return stats

def write_csv(stats: dict, output_path: Path):
    """Write statistics to CSV file"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Source', 'Rule1 Only', 'Rule2 Only', 'Both', 'Neither'])
        
        # Write stats for each source including overall
        for source in stats:
            writer.writerow([
                source,
                stats[source]['rule1_only'],
                stats[source]['rule2_only'],
                stats[source]['both'],
                stats[source]['neither']
            ])

def main():
    # Process all json files in the model_response directory
    model_response_dir = Path("model_response")
    output_dir = Path("/scratch/gpfs/lh2046/rules_conflicts/eval/eval_results")
    output_dir.mkdir(exist_ok=True)
    
    for filepath in model_response_dir.glob("*.json"):
        stats = process_file(filepath, output_dir)
        output_path = output_dir / f"{filepath.stem}_eval.csv"
        write_csv(stats, output_path)

if __name__ == "__main__":
    main()
