import json
from pathlib import Path
from collections import defaultdict

def analyze_connectivity_by_type():
    base_dir = Path(__file__).parent.parent
    qa_file = base_dir / "HotpotQA" / "hotpotqa_sample_200.json"
    links_file = base_dir / "question_supporting_links.json"

    print(f"Loading QA Data from {qa_file}...")
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    print(f"Loading Links Data from {links_file}...")
    with open(links_file, 'r', encoding='utf-8') as f:
        links_data = json.load(f)

    # Map question_id to type
    id_to_type = {}
    for item in qa_data:
        id_to_type[item['_id']] = item['type']

    # Map question_id to connectivity status
    # We check if 'shared_metadata_among_supporting' is not empty
    id_to_connected = {}
    for item in links_data:
        q_id = item['question_id']
        is_connected = len(item['shared_metadata_among_supporting']) > 0
        id_to_connected[q_id] = is_connected

    # Analysis
    stats = {
        "bridge": {"total": 0, "connected": 0},
        "comparison": {"total": 0, "connected": 0},
        "unknown": {"total": 0, "connected": 0}
    }

    for q_id, q_type in id_to_type.items():
        if q_id not in id_to_connected:
            continue # Skip if not in links analysis (shouldn't happen for sample)
        
        if q_type not in stats:
            q_type = "unknown"
            
        stats[q_type]["total"] += 1
        if id_to_connected[q_id]:
            stats[q_type]["connected"] += 1

    # Print Report
    print("\n=== Connectivity Analysis by Question Type ===")
    print(f"{'Type':<15} | {'Total':<10} | {'Connected':<10} | {'Ratio (%)':<10}")
    print("-" * 55)
    
    for q_type, data in stats.items():
        if data["total"] == 0:
            continue
        ratio = (data["connected"] / data["total"]) * 100
        print(f"{q_type:<15} | {data['total']:<10} | {data['connected']:<10} | {ratio:.2f}%")

if __name__ == "__main__":
    analyze_connectivity_by_type()
