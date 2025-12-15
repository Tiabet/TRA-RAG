import json
import os

def filter_qa_decomposition():
    failures_path = 'Analysis/recoverable_recall_failures_detailed.json'
    decomposition_path = 'Analysis/recoverable_qa_decomposition.json'
    
    print(f"Loading failure IDs from {failures_path}...")
    with open(failures_path, 'r', encoding='utf-8') as f:
        failures_data = json.load(f)
    
    failure_ids = set(item['id'] for item in failures_data)
    print(f"Found {len(failure_ids)} failure cases.")

    print(f"Loading decomposition data from {decomposition_path}...")
    with open(decomposition_path, 'r', encoding='utf-8') as f:
        decomposition_data = json.load(f)
    
    filtered_data = [item for item in decomposition_data if item['id'] in failure_ids]
    
    print(f"Filtered decomposition data from {len(decomposition_data)} to {len(filtered_data)} items.")

    print(f"Overwriting {decomposition_path} with filtered data...")
    with open(decomposition_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    filter_qa_decomposition()
