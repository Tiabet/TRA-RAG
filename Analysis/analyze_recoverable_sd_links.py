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
    output_json_path = os.path.join(base_dir, "Analysis", "recoverable_question_sd_links.json")

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
    
    total_ss_pairs = 0
    total_sd_pairs = 0

    print("Analyzing S-S and S-D links...")
    
    for item in recoverable_qa:
        q_text = item['question']
        q_id = item.get('id', 'unknown')
        
        # Get Supporting Facts (Titles)
        supporting_facts = set([fact[0] for fact in item.get('supporting_facts', [])])
        
        # Get Metadata for this question
        context_metadata_list = question_to_metadata.get(q_text)
        if not context_metadata_list:
            for mq, mdata in question_to_metadata.items():
                if mq.strip().lower() == q_text.strip().lower():
                    context_metadata_list = mdata
                    break
        
        if not context_metadata_list:
            print(f"Warning: No metadata found for {q_text[:30]}...")
            continue

        # Build Doc -> KV map for ALL docs in context
        doc_kvs = {}
        all_docs = set()
        
        for doc_meta in context_metadata_list:
            title = doc_meta.get('title')
            if not title: 
                title = doc_meta.get('metadata', {}).get('title')
            
            if not title: continue
            
            all_docs.add(title)
            
            meta = doc_meta.get('metadata', {})
            kvs = get_metadata_kv(meta)
            kvs.add(('self.title', normalize_text(title)))
            doc_kvs[title] = kvs

        # Identify S and D sets
        s_docs = [d for d in all_docs if d in supporting_facts]
        d_docs = [d for d in all_docs if d not in supporting_facts]
        
        # Analyze S-S Links
        ss_links_count = 0
        ss_shared_details = []

        # Iterate unique pairs
        for i in range(len(s_docs)):
            for j in range(i + 1, len(s_docs)):
                doc_a = s_docs[i]
                doc_b = s_docs[j]
                
                vals_a = set(v for k, v in doc_kvs[doc_a])
                vals_b = set(v for k, v in doc_kvs[doc_b])
                
                shared = vals_a.intersection(vals_b)
                for val in shared:
                    keys_a = [k for k, v in doc_kvs[doc_a] if v == val]
                    keys_b = [k for k, v in doc_kvs[doc_b] if v == val]
                    
                    link_weight = len(keys_a) + len(keys_b)
                    ss_links_count += link_weight

                    ss_shared_details.append({
                        "value": val,
                        "doc_a": doc_a,
                        "doc_b": doc_b,
                        "keys_a": keys_a,
                        "keys_b": keys_b,
                        "weight": link_weight
                    })

        # Analyze S-D Links
        sd_links_count = 0
        sd_shared_details = []
        
        for s_doc in s_docs:
            for d_doc in d_docs:
                vals_s = set(v for k, v in doc_kvs[s_doc])
                vals_d = set(v for k, v in doc_kvs[d_doc])
                
                shared = vals_s.intersection(vals_d)
                
                for val in shared:
                    # Find keys for this value
                    keys_s = [k for k, v in doc_kvs[s_doc] if v == val]
                    keys_d = [k for k, v in doc_kvs[d_doc] if v == val]
                    
                    link_weight = len(keys_s) + len(keys_d)
                    sd_links_count += link_weight
                    
                    sd_shared_details.append({
                        "value": val,
                        "s_doc": s_doc,
                        "d_doc": d_doc,
                        "s_keys": keys_s,
                        "d_keys": keys_d,
                        "weight": link_weight
                    })

        total_ss_pairs += ss_links_count
        total_sd_pairs += sd_links_count

        # Group S-S details by value
        grouped_ss_details = defaultdict(list)
        ss_value_weights = defaultdict(int)

        for detail in ss_shared_details:
            val = detail['value']
            ss_value_weights[val] += detail['weight']
            grouped_ss_details[val].append({
                "doc_a": detail['doc_a'],
                "doc_b": detail['doc_b'],
                "keys_a": detail['keys_a'],
                "keys_b": detail['keys_b'],
                "weight": detail['weight']
            })

        formatted_ss_details = []
        for val, links in grouped_ss_details.items():
            formatted_ss_details.append({
                "value": val,
                "count": ss_value_weights[val],
                "pair_count": len(links),
                "links": links
            })
        formatted_ss_details.sort(key=lambda x: x['count'], reverse=True)

        # Group S-D details by value to make JSON readable
        grouped_sd_details = defaultdict(list)
        value_weights = defaultdict(int)
        
        for detail in sd_shared_details:
            val = detail['value']
            value_weights[val] += detail['weight']
            grouped_sd_details[val].append({
                "s_doc": detail['s_doc'],
                "d_doc": detail['d_doc'],
                "s_keys": detail['s_keys'],
                "d_keys": detail['d_keys'],
                "weight": detail['weight']
            })
            
        formatted_sd_details = []
        for val, links in grouped_sd_details.items():
            formatted_sd_details.append({
                "value": val,
                "count": value_weights[val], # Total weight for this value across all pairs
                "pair_count": len(links),
                "links": links
            })
            
        # Sort by count desc
        formatted_sd_details.sort(key=lambda x: x['count'], reverse=True)

        results.append({
            "question_id": q_id,
            "question": q_text,
            "stats": {
                "num_supporting_docs": len(s_docs),
                "num_distractor_docs": len(d_docs),
                "ss_link_key_count": ss_links_count,
                "sd_link_key_count": sd_links_count,
                "ratio_sd_to_ss": sd_links_count / ss_links_count if ss_links_count > 0 else 0
            },
            "ss_shared_values": formatted_ss_details,
            "sd_shared_values": formatted_sd_details
        })

    print(f"\nTotal S-S Links (Key Count): {total_ss_pairs}")
    print(f"Total S-D Links (Key Count): {total_sd_pairs}")
    if total_ss_pairs > 0:
        print(f"Overall Ratio (S-D / S-S): {total_sd_pairs / total_ss_pairs:.2f}")

    print(f"Saving detailed results to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
