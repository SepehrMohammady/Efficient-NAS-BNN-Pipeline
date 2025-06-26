import torch
from utils import tuple2cand # Assuming cand2tuple and tuple2cand are in utils

info_path = './work_dirs/cifar10_nasbnn_exp/search/info.pth.tar'
search_results = torch.load(info_path, map_location='cpu')

print("--- Pareto Global Architectures ---")
pareto = search_results.get('pareto_global', {})
vis_dict = search_results.get('vis_dict', {})

if not pareto:
    print("Pareto front is empty.")
else:
    # Sort by OPs for display
    # The keys in pareto_global are the OPs bucket indices (e.g., 0, 1, 2... representing ops_step*idx)
    # Or they might be the float OPs if step was small. Let's assume they are sortable.
    
    # The OPs value from the pareto front log was an integer bucket, 
    # but the step was 0.05. Let's just print what's there.
    for ops_bucket_key in sorted(pareto.keys()):
        cand_tuple = pareto[ops_bucket_key]
        if cand_tuple in vis_dict:
            acc = vis_dict[cand_tuple]['acc']
            ops_val = vis_dict[cand_tuple]['ops']
            print(f"OPs Bucket Key: {ops_bucket_key} (Actual OPs: {ops_val:.4f}M) -> Accuracy: {acc:.2f}%, Arch: {cand_tuple}")
        else:
            print(f"OPs Bucket Key: {ops_bucket_key} -> Arch: {cand_tuple} (Details not in vis_dict?)")

# You can also explore vis_dict for other candidates if needed
# print("\n--- Sample from Visited Dictionary (vis_dict) ---")
# count = 0
# for cand_t, data in vis_dict.items():
#     print(f"Arch: {cand_t}, Acc: {data.get('acc', 'N/A')}, OPs: {data.get('ops', 'N/A')}")
#     count += 1
#     if count > 5: # Print first 5
#         break