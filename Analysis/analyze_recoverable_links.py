import json
import os
from collections import defaultdict, Counter

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # Paths
    base_dir = r"c:\Development\ChunkRAG_v2"
    # Input: The filtered QA file containing only "Recoverable" questions
    recoverable_qa_path = os.path.join(base_dir, "Analysis", "recoverable_musique_qa.json")
    # Metadata: The full metadata file
    metadata_path = os.path.join(base_dir, "MuSiQue", "musique_sample_200_metadata.json")

    print(f"Loading Recoverable QA Data from {recoverable_qa_path}...")
    recoverable_qa = load_json(recoverable_qa_path)
    print(f"Found {len(recoverable_qa)} recoverable questions.")

    # 3. Load Metadata
    print("Loading metadata...")
    metadata_list = load_json(metadata_path)
    question_to_metadata = {}
    for item in metadata_list:
        q_text = item['question']
        context_meta = {}
        for cm in item['context_metadata']:
            context_meta[cm['title']] = cm['metadata']
        question_to_metadata[q_text] = context_meta

    # Analysis Stats
    ss_link_types = Counter()
    sd_link_types = Counter()
    
    ss_link_values = Counter()
    sd_link_values = Counter()

    print("Analyzing links for Recoverable Questions...")
    
    processed_count = 0
    for sample_item in recoverable_qa:
        q_text = sample_item['question']
        
        supporting_titles = set(fact[0] for fact in sample_item.get('supporting_facts', []))
        
        if q_text not in question_to_metadata:
            # Try normalizing?
            # The metadata file might have slightly different whitespace/formatting
            # But usually they match if from same source.
            # Let's try simple normalization if direct match fails
            found = False
            for mq in question_to_metadata:
                if mq.strip().lower() == q_text.strip().lower():
                    context_meta = question_to_metadata[mq]
                    found = True
                    break
            if not found:
                print(f"Warning: Metadata not found for question: {q_text[:50]}...")
                continue
        else:
            context_meta = question_to_metadata[q_text]
        
        processed_count += 1

        # Helper to get (key, value) pairs
        def get_metadata_kv(meta_obj, parent_key=""):
            kv_pairs = set()
            if isinstance(meta_obj, dict):
                for k, v in meta_obj.items():
                    if k == 'title': continue
                    current_key = f"{parent_key}.{k}" if parent_key else k
                    kv_pairs.update(get_metadata_kv(v, current_key))
            elif isinstance(meta_obj, list):
                for item in meta_obj:
                    kv_pairs.update(get_metadata_kv(item, parent_key)) # Keep parent key for list items
            elif isinstance(meta_obj, (str, int, float, bool)):
                kv_pairs.add((parent_key, str(meta_obj)))
            return kv_pairs

        # Pre-calculate KV for all documents
        doc_kv = {}
        for title, meta in context_meta.items():
            doc_kv[title] = get_metadata_kv(meta)

        # Analyze links
        for supp_title in supporting_titles:
            if supp_title not in doc_kv: continue
            
            supp_kvs = doc_kv[supp_title]
            
            for other_title, other_kvs in doc_kv.items():
                if supp_title == other_title: continue
                
                # Find shared VALUES (regardless of key, but we track the key of the source)
                
                supp_values = {v: k for k, v in supp_kvs}
                other_values = {v: k for k, v in other_kvs}
                
                shared_vals = set(supp_values.keys()).intersection(set(other_values.keys()))
                
                for val in shared_vals:
                    key_in_supp = supp_values[val]
                    key_in_other = other_values[val]
                    
                    link_type = f"{key_in_supp} <-> {key_in_other}"
                    
                    if other_title in supporting_titles:
                        ss_link_types[link_type] += 1
                        ss_link_values[val] += 1
                    else:
                        sd_link_types[link_type] += 1
                        sd_link_values[val] += 1

    # Report
    print("\n" + "="*50)
    print(f"RECOVERABLE ERROR ANALYSIS ({processed_count} questions)")
    print("COMPARATIVE ANALYSIS: Supporting-Supporting (S-S) vs Supporting-Distractor (S-D) Links")
    print("="*50)
    
    print("\nTop 10 Metadata Key Patterns in S-S Links:")
    for k, v in ss_link_types.most_common(10):
        print(f"  {k}: {v}")
        
    print("\nTop 10 Metadata Key Patterns in S-D Links:")
    for k, v in sd_link_types.most_common(10):
        print(f"  {k}: {v}")

    print("\nTop 10 Shared Values in S-S Links:")
    for k, v in ss_link_values.most_common(10):
        print(f"  {k}: {v}")

    print("\nTop 10 Shared Values in S-D Links:")
    for k, v in sd_link_values.most_common(10):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
