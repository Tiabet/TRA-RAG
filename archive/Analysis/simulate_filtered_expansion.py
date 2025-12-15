import json
import os
from collections import defaultdict

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
    results_path = os.path.join(base_dir, "Results", "test_musique_v4_200_results.json")
    metadata_path = os.path.join(base_dir, "MuSiQue", "musique_sample_200_metadata.json")
    gold_path = os.path.join(base_dir, "MuSiQue", "musique_sample_200.json")
    
    print("Loading Data...")
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    with open(gold_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
        
    q_to_gold = {}
    for item in gold_data:
        q_text = normalize_text(item['question'])
        facts = [fact[0] for fact in item.get('supporting_facts', [])]
        q_to_gold[q_text] = facts

    # --- STRATEGY CONFIGURATION ---
    STOP_VALUES = {
        'located_in', 'american', 'united states', 'part_of', 'true', 'false', 'written_by',
        '2', '3', '5', '1', '4', '6', '7', '8', '9', '10', '0'
    }
    
    def is_valid_key(k):
        # Block generic relation names
        if k.endswith('relation'): return False
        return True

    def is_valid_value(v):
        if v in STOP_VALUES: return False
        return True
    # ------------------------------

    print("Building Metadata Graph (With Filters)...")
    value_to_docs = defaultdict(set)
    doc_to_values = defaultdict(set)
    
    for item in metadata_list:
        context_meta = item.get('context_metadata', [])
        for doc_meta in context_meta:
            title = doc_meta.get('title')
            if not title:
                title = doc_meta.get('metadata', {}).get('title')
            
            if not title: continue
            
            norm_title = normalize_text(title)
            
            meta = doc_meta.get('metadata', {})
            kvs = get_metadata_kv(meta)
            
            # Add title itself
            kvs.add(('self.title', norm_title))
            
            for k, v in kvs:
                # APPLY FILTERS HERE
                if not is_valid_key(k): continue
                if not is_valid_value(v): continue
                
                value_to_docs[v].add(norm_title)
                doc_to_values[norm_title].add(v)

    print("Simulating Filtered Link Expansion...")
    
    original_metrics = {'recall': [], 'precision': [], 'f1': [], 'hit': []}
    expanded_metrics = {'recall': [], 'precision': [], 'f1': [], 'hit': []}
    
    avg_expansion_count = 0
    
    for item in results_data['results']:
        q_text = normalize_text(item['question'])
        gold_facts = q_to_gold.get(q_text)
        
        if not gold_facts:
            continue
            
        retrieved_titles = set()
        if 'decomposition' in item and 'subquestions' in item['decomposition']:
            for sq in item['decomposition']['subquestions']:
                for p in sq.get('retrieved_passages', []):
                    if 'title' in p:
                        retrieved_titles.add(normalize_text(p['title']))
        
        r, p, f, h = calculate_metrics(retrieved_titles, gold_facts)
        original_metrics['recall'].append(r)
        original_metrics['precision'].append(p)
        original_metrics['f1'].append(f)
        original_metrics['hit'].append(h)
        
        # Expand
        expanded_titles = set(retrieved_titles)
        
        for title in retrieved_titles:
            my_values = doc_to_values.get(title, set())
            for val in my_values:
                neighbors = value_to_docs.get(val, set())
                expanded_titles.update(neighbors)
        
        avg_expansion_count += len(expanded_titles)
        
        r_ex, p_ex, f_ex, h_ex = calculate_metrics(expanded_titles, gold_facts)
        expanded_metrics['recall'].append(r_ex)
        expanded_metrics['precision'].append(p_ex)
        expanded_metrics['f1'].append(f_ex)
        expanded_metrics['hit'].append(h_ex)

    avg_expansion_count /= len(results_data['results'])

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print("\n" + "="*50)
    print("FILTERED RETRIEVAL SIMULATION REPORT")
    print("="*50)
    print(f"Filters Applied:")
    print(f"1. Key Filter: Block *.relation")
    print(f"2. Value Filter: Block {len(STOP_VALUES)} generic terms")
    print(f"Avg Expanded Docs per Query: {avg_expansion_count:.1f}")
    
    print("\n[Baseline: Current Retrieval]")
    print(f"Avg Recall:    {avg(original_metrics['recall']):.4f}")
    print(f"Avg Precision: {avg(original_metrics['precision']):.4f}")
    print(f"Avg F1:        {avg(original_metrics['f1']):.4f}")
    print(f"Hit Rate (All):{avg(original_metrics['hit']):.4f}")
    
    print("\n[Simulation: Filtered Expansion]")
    print(f"Avg Recall:    {avg(expanded_metrics['recall']):.4f}")
    print(f"Avg Precision: {avg(expanded_metrics['precision']):.4f}")
    print(f"Avg F1:        {avg(expanded_metrics['f1']):.4f}")
    print(f"Hit Rate (All):{avg(expanded_metrics['hit']):.4f}")
    
    print("\n[Impact]")
    print(f"Recall Gain:   {avg(expanded_metrics['recall']) - avg(original_metrics['recall']):.4f}")
    print(f"Precision Drop:{avg(original_metrics['precision']) - avg(expanded_metrics['precision']):.4f}")

if __name__ == "__main__":
    main()
