import json
import os

def extract_recoverable_results():
    # Paths
    recoverable_qa_path = 'Analysis/recoverable_musique_qa.json'
    full_results_path = 'Results/test_musique_v4_200_results.json'
    output_path = 'Analysis/recoverable_musique_results.json'

    # Load recoverable QAs
    print(f"Loading recoverable QAs from {recoverable_qa_path}...")
    with open(recoverable_qa_path, 'r', encoding='utf-8') as f:
        recoverable_qas = json.load(f)
    
    recoverable_ids = set(item['_id'] for item in recoverable_qas)
    print(f"Found {len(recoverable_ids)} recoverable questions.")

    # Load full results
    print(f"Loading full results from {full_results_path}...")
    with open(full_results_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    if isinstance(full_data, dict) and 'results' in full_data:
        full_results = full_data['results']
    else:
        full_results = full_data

    print(f"Found {len(full_results)} total results.")

    # Filter results
    # Note: recoverable_qas uses '_id', full_results uses 'id'
    recoverable_results = [res for res in full_results if res.get('id') in recoverable_ids]
    print(f"Extracted {len(recoverable_results)} results matching recoverable IDs.")

    # Save extracted results
    print(f"Saving extracted results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(recoverable_results, f, indent=2, ensure_ascii=False)
    
    print("Done.")

if __name__ == "__main__":
    extract_recoverable_results()
