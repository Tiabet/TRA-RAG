import json
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm

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

def main():
    base_dir = r"c:\Development\ChunkRAG_v2"
    results_path = os.path.join(base_dir, "Results", "TMP_V3", "test_musique_v3_200_results_1.json")
    metadata_path = os.path.join(base_dir, "MuSiQue", "musique_sample_200_metadata.json")
    gold_path = os.path.join(base_dir, "MuSiQue", "musique_sample_200.json")
    output_path = os.path.join(base_dir, "Analysis", "expanded_sd_links.json")
    
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
        
    # Build Metadata Graph (Value -> Docs) & (Doc -> Values)
    print("Building Metadata Graph...")
    value_to_docs = defaultdict(set)
    doc_to_values = defaultdict(list) # Store (key, value) tuples
    
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
                # We link by Value only (Shared Value)
                value_to_docs[v].add(title)
                doc_to_values[title].append((k, v))

    print(f"Graph built. {len(value_to_docs)} unique values.")

    analysis_results = []
    
    print("Analyzing Expanded Sets...")
    for res in tqdm(results_data['results']):
        qid = res['id']
        if qid not in gold_map:
            continue
            
        gold_item = gold_map[qid]
        gold_supporting_titles = set(fact[0] for fact in gold_item.get('supporting_facts', []))
        
        # 1. Get Initial Retrieved Titles
        initial_titles = set()
        if 'decomposition' in res and 'subquestions' in res['decomposition']:
            for sq in res['decomposition']['subquestions']:
                if 'retrieved_passages' in sq and sq['retrieved_passages']:
                    for p in sq['retrieved_passages']:
                        initial_titles.add(p['title'])
        
        if not initial_titles:
            continue

        # 2. Expand (Shared Value)
        expanded_titles = set(initial_titles)
        for title in initial_titles:
            # Get all values for this doc
            kvs = doc_to_values.get(title, [])
            for k, v in kvs:
                # Get all docs sharing this value
                linked_docs = value_to_docs.get(v, set())
                expanded_titles.update(linked_docs)
        
        # 3. Define Sets
        # Source: Initial Retrieved Titles (R)
        r_docs = list(initial_titles)
        
        # Targets: Expanded Titles (E), split into Support (S) and Distractor (D)
        s_expanded_docs = []
        d_expanded_docs = []
        
        for title in expanded_titles:
            if title in gold_supporting_titles:
                s_expanded_docs.append(title)
            else:
                d_expanded_docs.append(title)
                
        # 4. Analyze Links
        # We want to find shared values between R and S (RS Links) and R and D (RD Links)
        
        # Helper to find shared values between two sets of docs
        def find_shared_values(source_group, target_group, src_label, tgt_label, src_keys_label, tgt_keys_label):
            shared_value_counts = defaultdict(lambda: {'count': 0, 'pair_count': 0, 'links': []})
            
            # Iterate over Source Group (Retrieved)
            for doc_a in source_group:
                kvs_a = doc_to_values.get(doc_a, [])
                
                # Iterate over Target Group (Expanded S or D)
                for doc_b in target_group:
                    if doc_a == doc_b: continue # Skip self-links
                    
                    kvs_b = doc_to_values.get(doc_b, [])
                    
                    # Map value -> keys for doc_b
                    val_map_b = defaultdict(list)
                    for k, v in kvs_b:
                        val_map_b[v].append(k)
                        
                    for k_a, v_a in kvs_a:
                        if v_a in val_map_b:
                            # Match found!
                            keys_b = val_map_b[v_a]
                            
                            entry = shared_value_counts[v_a]
                            entry['count'] += 1 
                            
                            entry['links'].append({
                                src_label: doc_a, # Source (Retrieved)
                                tgt_label: doc_b, # Target (Expanded)
                                src_keys_label: [k_a],
                                tgt_keys_label: keys_b,
                                'weight': 1
                            })
            
            # Post-process
            final_list = []
            for val, data in shared_value_counts.items():
                links = data['links']
                unique_pairs = set()
                for l in links:
                    pair = (l[src_label], l[tgt_label])
                    unique_pairs.add(pair)
                
                data['pair_count'] = len(unique_pairs)
                data['value'] = val
                final_list.append(data)
                
            # Sort by pair_count desc
            final_list.sort(key=lambda x: x['pair_count'], reverse=True)
            return final_list

        # RS Links (Retrieved -> Support)
        rs_shared = find_shared_values(r_docs, s_expanded_docs, 'r_doc', 's_doc', 'r_keys', 's_keys')
        
        # RD Links (Retrieved -> Distractor)
        rd_shared = find_shared_values(r_docs, d_expanded_docs, 'r_doc', 'd_doc', 'r_keys', 'd_keys')
        
        # Identify Direct Hits (Retrieved docs that are also Supporting)
        direct_hits = [doc for doc in r_docs if doc in gold_supporting_titles]
        
        analysis_results.append({
            "question_id": qid,
            "question": res['question'],
            "stats": {
                "num_retrieved_docs": len(r_docs),
                "num_expanded_support_docs": len(s_expanded_docs),
                "num_expanded_distractor_docs": len(d_expanded_docs),
                "num_direct_hits": len(direct_hits),
                "rs_link_key_count": len(rs_shared),
                "rd_link_key_count": len(rd_shared),
                "ratio_rd_to_rs": len(rd_shared) / len(rs_shared) if len(rs_shared) > 0 else 0
            },
            "direct_hits": direct_hits,
            "rs_shared_values": rs_shared,
            "rd_shared_values": rd_shared
        })

    print(f"Saving analysis to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
