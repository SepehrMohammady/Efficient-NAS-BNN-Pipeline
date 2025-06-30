import torch
import os
from utils import tuple2cand # Assuming cand2tuple and tuple2cand are in utils

# Default path - change as needed
info_path = './work_dirs/wakevision_nasbnn_LARGEXP_run/search/info.pth.tar'

# Check if file exists and prompt for alternative if not found
if not os.path.exists(info_path):
    print(f"Error: File {info_path} not found.")
    user_input = input("Enter the path to your info.pth.tar file: ")
    if os.path.exists(user_input):
        info_path = user_input
    else:
        print("The specified file does not exist. Please provide a valid path.")
        exit(1)

try:
    search_results = torch.load(info_path, map_location='cpu')
    print(f"Successfully loaded search results from {info_path}")
except Exception as e:
    print(f"Error loading search results: {e}")
    exit(1)

print("\n--- Pareto Global Architectures ---")
pareto = search_results.get('pareto_global', {})
vis_dict = search_results.get('vis_dict', {})

print(f"Total number of evaluated architectures in vis_dict: {len(vis_dict)}")
print(f"Number of architectures on the Pareto front: {len(pareto)}")

if not pareto:
    print("Pareto front is empty.")
else:
    # Sort by OPs for display
    for ops_bucket_key in sorted(pareto.keys()):
        cand_tuple = pareto[ops_bucket_key]
        if cand_tuple in vis_dict:
            acc = vis_dict[cand_tuple]['acc']
            ops_val = vis_dict[cand_tuple]['ops']
            print(f"OPs Bucket Key: {ops_bucket_key} (Actual OPs: {ops_val:.4f}M) -> Accuracy: {acc:.2f}%, Arch: {cand_tuple}")
        else:
            print(f"OPs Bucket Key: {ops_bucket_key} -> Arch: {cand_tuple} (Details not in vis_dict?)")

print("\n--- Sample from Visited Dictionary (vis_dict) ---")
count = 0
sample_count = min(5, len(vis_dict))
for cand_t, data in list(vis_dict.items())[:sample_count]:
    print(f"Arch: {cand_t}, Acc: {data.get('acc', 'N/A')}, OPs: {data.get('ops', 'N/A')}")
    count += 1

# Print summary statistics of search epochs
print("\n--- Search Progress Summary ---")
memory = search_results.get('memory', [])
print(f"Total epochs completed: {search_results.get('epoch', 0)}")
print(f"Length of memory array: {len(memory)}")

# Optionally export the Pareto front architectures to a CSV file
try:
    import pandas as pd
    pareto_data = []
    for ops_key, cand_tuple in pareto.items():
        if cand_tuple in vis_dict:
            pareto_data.append({
                'OPs_Bucket': ops_key,
                'Accuracy': vis_dict[cand_tuple]['acc'],
                'OPs': vis_dict[cand_tuple]['ops'],
                'Architecture': str(cand_tuple)
            })
    
    if pareto_data:
        df = pd.DataFrame(pareto_data)
        csv_path = os.path.join(os.path.dirname(info_path), 'pareto_architectures.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nPareto architectures exported to {csv_path}")
except ImportError:
    print("\nPandas not available, skipping CSV export")
except Exception as e:
    print(f"\nError exporting Pareto architectures to CSV: {e}")
