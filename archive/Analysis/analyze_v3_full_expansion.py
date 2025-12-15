import json
import os
from collections import defaultdict
import numpy as np

def normalize_text(text):
    if text is None:
        return ""
    s = str(text).lower()
    s = s.replace(',', '')
    return s.strip()

def get_metadata_kv(meta_obj, parent_key=""):
    kv_pairs = set()
    if isinstance(meta_obj, dict):
        for k, v in meta_obj.items():
            if k == 'title': continue
            current_key = f"{parent_key}.{k}" if parent_key else k
            kv_pairs.update(get_metadata_kv(v, current_key))
    elif isinstance(meta_obj, list):
        for item in meta_obj:
            kv_pairs.update(get_metadata_kv(item, parent_key))
    elif isinstance(meta_obj, (str, int, float, bool)):
        val = normalize_text(meta_obj)
        if val:
            kv_pairs.add((parent_key, val))
    return kv_pairs

def calculate_metrics(retrieved_docs, gold_docs):
    if not gold_docs:
        return 0.0, 0.0, 0.0, 0.0
    
    retrieved_set = set(normalize_text(d) for d in retrieved_docs)
    gold_set = set(normalize_text(d) for d in gold_docs)
    
    intersection = retrieved_set.intersection(gold_set)
    
    recall = len(intersection) / len(gold_set) if gold_set else 0.0
    precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    hit = 1.0 if len(intersection) == len(gold_set) else 0.0
    
    return recall, precision, f1, hit

def main():
    base_dir = r"c:\Development\ChunkRAG_v2"
    results_path = os.path.join(base_dir, "Results", "TMP_V3", "test_musique_v3_200_results_1.json")
    metadata_path = os.path.join(base_dir, "MuSiQue", "musique_sample_200_metadata.json")
    gold_path = os.path.join(base_dir, "MuSiQue", "musique_sample_200.json")
    
    print(f"Loading Results from: {results_path}")
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
        
    print(f"Loading Metadata from: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
        
    print(f"Loading Gold Data from: {gold_path}")
    with open(gold_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
        
    # Map gold data by ID
    gold_map = {item.get('_id', item.get('id')): item for item in gold_data}
        
    # Build Metadata Graph (Value -> Docs)
    print("Building Metadata Graph...")
    value_to_docs = defaultdict(set)
    doc_to_values = defaultdict(set)
    
    for item in metadata_list:
        context_meta = item.get('context_metadata', [])
        for doc_meta in context_meta:
            title = doc_meta.get('title')
            if not title:
                title = doc_meta.get('metadata', {}).get('title')
            
            if not title:
                continue
                
            # Extract all values
            kv_pairs = get_metadata_kv(doc_meta.get('metadata', {}))
            for k, v in kv_pairs:
                # We use (key, value) as the unique identifier for a value node
                # to avoid mixing up same values in different fields (though user said "value based", usually implies shared value)
                # But strict value sharing might be too broad if we ignore keys. 
                # Let's try strict (key, value) matching first as it's safer, 
                # or just value matching if that's what "Shared Value" usually means in this project.
                # Based on previous context, "Shared Value" usually implies the value itself matches.
                # Let's use just the value `v` for broader expansion as "Full Expansion" implies maximum recall.
                
                # Actually, let's stick to (key, value) to be precise, or just value?
                # In `metadata_graph.py` (if it exists), we usually link by value.
                # Let's use just `v` to be consistent with "Shared Value" linking.
                
                value_to_docs[v].add(title)
                doc_to_values[title].add(v)

    print(f"Graph built. {len(value_to_docs)} unique values.")

    # Analyze
    print("\nAnalyzing Retrieval Performance with Full Expansion...")
    
    metrics = {
        'initial': {'recall': [], 'precision': [], 'f1': [], 'hit': []},
        'expanded': {'recall': [], 'precision': [], 'f1': [], 'hit': []}
    }
    
    expansion_stats = []
    
    for res in results_data['results']:
        qid = res['id']
        if qid not in gold_map:
            continue
            
        gold_item = gold_map[qid]
        # gold_paragraphs = [p['title'] for p in gold_item['paragraphs'] if p['is_supporting']]
        gold_paragraphs = set(fact[0] for fact in gold_item.get('supporting_facts', []))
        
        # 1. Get Initial Retrieved Titles
        initial_titles = set()
        if 'decomposition' in res and 'subquestions' in res['decomposition']:
            for sq in res['decomposition']['subquestions']:
                if 'retrieved_passages' in sq and sq['retrieved_passages']:
                    for p in sq['retrieved_passages']:
                        initial_titles.add(p['title'])
        
        # 2. Expand
        expanded_titles = set(initial_titles)
        for title in initial_titles:
            # Get all values for this doc
            values = doc_to_values.get(title, set())
            for v in values:
                # Get all docs sharing this value
                linked_docs = value_to_docs.get(v, set())
                expanded_titles.update(linked_docs)
                
        # 3. Calculate Metrics
        # Initial
        i_rec, i_prec, i_f1, i_hit = calculate_metrics(initial_titles, gold_paragraphs)
        metrics['initial']['recall'].append(i_rec)
        metrics['initial']['precision'].append(i_prec)
        metrics['initial']['f1'].append(i_f1)
        metrics['initial']['hit'].append(i_hit)
        
        # Expanded
        e_rec, e_prec, e_f1, e_hit = calculate_metrics(expanded_titles, gold_paragraphs)
        metrics['expanded']['recall'].append(e_rec)
        metrics['expanded']['precision'].append(e_prec)
        metrics['expanded']['f1'].append(e_f1)
        metrics['expanded']['hit'].append(e_hit)
        
        expansion_stats.append({
            'initial_count': len(initial_titles),
            'expanded_count': len(expanded_titles),
            'factor': len(expanded_titles) / len(initial_titles) if initial_titles else 0
        })

    # Print Summary
    print("\n=== Summary Results ===")
    print(f"Total Questions Analyzed: {len(metrics['initial']['recall'])}")
    
    print("\n[Initial Retrieval]")
    print(f"Avg Recall:    {np.mean(metrics['initial']['recall']):.4f}")
    print(f"Avg Precision: {np.mean(metrics['initial']['precision']):.4f}")
    print(f"Avg F1:        {np.mean(metrics['initial']['f1']):.4f}")
    print(f"Hit Rate:      {np.mean(metrics['initial']['hit']):.4f}")
    
    print("\n[Full Expansion (Shared Value)]")
    print(f"Avg Recall:    {np.mean(metrics['expanded']['recall']):.4f}")
    print(f"Avg Precision: {np.mean(metrics['expanded']['precision']):.4f}")
    print(f"Avg F1:        {np.mean(metrics['expanded']['f1']):.4f}")
    print(f"Hit Rate:      {np.mean(metrics['expanded']['hit']):.4f}")
    
    avg_expansion = np.mean([s['factor'] for s in expansion_stats])
    avg_docs = np.mean([s['expanded_count'] for s in expansion_stats])
    print(f"\nAvg Expansion Factor: {avg_expansion:.2f}x")
    print(f"Avg Expanded Docs:    {avg_docs:.1f}")

if __name__ == "__main__":
    main()
