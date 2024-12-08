import pandas as pd
import numpy as np

# List of metrics
metrics = ['mmlu', 'math', 'gsm8k', 'bbh', 'human_eval', 'mt_bench', 'truthfulqa']

# Initialize a dictionary to store metric values for each seed
metric_values = {metric: [] for metric in metrics}

# Loop over each seed and read the corresponding CSV file
for seed in [1, 2, 3, 4, 5]:
    filename = f'llama3-8B-Instruct-TAR-Bio-{seed}-nosys+score.csv'
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        continue
    
    # For each metric, find its score in the 'bench' and 'score' columns
    for metric in metrics:
        # Check if the metric exists in the 'bench' column
        if metric in df['bench'].values:
            # Extract the score corresponding to the metric
            score = df.loc[df['bench'] == metric, 'score'].values[0]
            metric_values[metric].append(score)
        else:
            print(f"Metric '{metric}' not found in {filename}.")

# Compute mean and standard deviation for each metric
results = {}
for metric in metrics:
    values = metric_values[metric]
    if values:
        mean = np.mean(values)
        std = np.std(values)
        results[metric] = {'mean': mean, 'std': std}
    else:
        results[metric] = {'mean': None, 'std': None}

# Display the results
print("Metric Results:")
for metric in metrics:
    mean = results[metric]['mean']
    std = results[metric]['std']
    confidence_interval = 1.96 * std / np.sqrt(5)
    if mean is not None:
        print(f"{metric}: Mean = {mean:.4f}, Confidence_interval={confidence_interval: .4f}")
    else:
        print(f"{metric}: No data available.")