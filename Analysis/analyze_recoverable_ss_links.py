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

def main():
    base_dir = r"c:\Development\ChunkRAG_v2"
    recoverable_qa_path = os.path.join(base_dir, "Analysis", "recoverable_musique_qa.json")
    metadata_path = os.path.join(base_dir, "MuSiQue", "musique_sample_200_metadata.json")
    output_path = os.path.join(base_dir, "Analysis", "recoverable_question_ss_links.json")

    print(f"Loading Recoverable QA from {recoverable_qa_path}...")
    with open(recoverable_qa_path, 'r', encoding='utf-8') as f:
        recoverable_qa = json.load(f)

    print(f"Loading Metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    
    # Map question to metadata
    question_to_metadata = {}
    for item in metadata_list:
        question_to_metadata[item['question']] = item['context_metadata']

    results = []

    print("Analyzing S-S links...")
    for item in recoverable_qa:
        q_text = item['question']
        q_id = item.get('id', 'unknown')
        
        # Get Supporting Facts (Titles)
        supporting_facts = [fact[0] for fact in item.get('supporting_facts', [])]
        supporting_facts_set = set(supporting_facts)
        
        # Get Metadata for this question
        # Try exact match first, then normalized
        context_metadata_list = question_to_metadata.get(q_text)
        if not context_metadata_list:
            # Fallback search
            for mq, mdata in question_to_metadata.items():
                if mq.strip().lower() == q_text.strip().lower():
                    context_metadata_list = mdata
                    break
        
        if not context_metadata_list:
            print(f"Warning: No metadata found for {q_text[:30]}...")
            continue

        # Build Doc -> KV map for Supporting Facts ONLY
        doc_kvs = {}
        for doc_meta in context_metadata_list:
            title = doc_meta.get('title')
            if not title: 
                title = doc_meta.get('metadata', {}).get('title')
            
            if title in supporting_facts_set:
                # Extract KVs
                meta = doc_meta.get('metadata', {})
                
                # Add title itself as a value (self.title)
                kvs = get_metadata_kv(meta)
                kvs.add(('self.title', normalize_text(title)))
                
                doc_kvs[title] = kvs

        # Find Shared Values among Supporting Facts
        # Value -> List of (Doc, Key)
        value_map = defaultdict(list)
        
        for title, kvs in doc_kvs.items():
            for key, val in kvs:
                value_map[val].append((title, key))

        # Filter for values shared by at least 2 docs
        shared_metadata_among_supporting = []
        
        for val, occurrences in value_map.items():
            if len(occurrences) > 1:
                # It's a link!
                docs = sorted(list(set(occ[0] for occ in occurrences)))
                keys = sorted(list(set(occ[1] for occ in occurrences)))
                
                # Only count if it links distinct documents
                if len(docs) > 1:
                    shared_metadata_among_supporting.append({
                        "value": val,
                        "keys": keys,
                        "docs": docs
                    })

        results.append({
            "question_id": q_id,
            "question": q_text,
            "supporting_facts": supporting_facts,
            "shared_metadata_among_supporting": shared_metadata_among_supporting
        })

    print(f"Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
