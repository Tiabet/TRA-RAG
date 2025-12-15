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

def main():
    base_dir = r"c:\Development\ChunkRAG_v2"
    results_path = os.path.join(base_dir, "Results", "test_musique_v4_200_results.json")
    metadata_path = os.path.join(base_dir, "MuSiQue", "musique_sample_200_metadata.json")
    
    print("Loading Data...")
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
        
    # Build Metadata Graph
    print("Building Metadata Graph...")
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
                value_to_docs[v].add(norm_title)
                doc_to_values[norm_title].add(v)

    print("Analyzing Expansion Stats...")
    
    sq_counts = []
    query_counts = []
    
    # If results_data is a dict with 'results' key
    if isinstance(results_data, dict) and 'results' in results_data:
        results_list = results_data['results']
    else:
        results_list = results_data

    for item in results_list:
        query_expanded_docs = set()
        
        if 'decomposition' in item and 'subquestions' in item['decomposition']:
            for sq in item['decomposition']['subquestions']:
                sq_retrieved_titles = set()
                for p in sq.get('retrieved_passages', []):
                    if 'title' in p:
                        sq_retrieved_titles.add(normalize_text(p['title']))
                
                # Expand SQ
                sq_expanded_titles = set(sq_retrieved_titles)
                for title in sq_retrieved_titles:
                    my_values = doc_to_values.get(title, set())
                    for val in my_values:
                        neighbors = value_to_docs.get(val, set())
                        sq_expanded_titles.update(neighbors)
                
                sq_counts.append(len(sq_expanded_titles))
                query_expanded_docs.update(sq_expanded_titles)
        
        # Fallback if no decomposition or empty
        if not query_expanded_docs and 'retrieved_passages' in item:
             # Treat as single SQ
             sq_retrieved_titles = set()
             for p in item.get('retrieved_passages', []):
                 if 'title' in p:
                     sq_retrieved_titles.add(normalize_text(p['title']))
             
             sq_expanded_titles = set(sq_retrieved_titles)
             for title in sq_retrieved_titles:
                 my_values = doc_to_values.get(title, set())
                 for val in my_values:
                     neighbors = value_to_docs.get(val, set())
                     sq_expanded_titles.update(neighbors)
             
             sq_counts.append(len(sq_expanded_titles))
             query_expanded_docs.update(sq_expanded_titles)

        query_counts.append(len(query_expanded_docs))

    print("\n" + "="*50)
    print("FULL EXPANSION STATISTICS")
    print("="*50)
    print(f"Total Queries Analyzed: {len(query_counts)}")
    print(f"Total SQs Analyzed:     {len(sq_counts)}")
    print("-" * 30)
    print(f"Avg Passages per SQ:    {np.mean(sq_counts):.2f} (Median: {np.median(sq_counts):.2f})")
    print(f"Max Passages per SQ:    {np.max(sq_counts)}")
    print("-" * 30)
    print(f"Avg Passages per Query: {np.mean(query_counts):.2f} (Median: {np.median(query_counts):.2f})")
    print(f"Max Passages per Query: {np.max(query_counts)}")
    print("="*50)

if __name__ == "__main__":
    main()
