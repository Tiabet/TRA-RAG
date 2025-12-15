import json
from pathlib import Path
from collections import defaultdict

def normalize_text(text):
    if text is None:
        return ""
    s = str(text).lower()
    s = s.replace(',', '')
    return s.strip()

def flatten_metadata(meta):
    """
    Recursively extracts (key_path, value) pairs from metadata.
    """
    items = []
    
    def recurse(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                recurse(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item, path)
        else:
            val = normalize_text(obj)
            if val:
                items.append((path, val))

    recurse(meta.get('attributes', {}), 'attributes')
    
    for rel in meta.get('relations', []):
        if isinstance(rel, dict):
            pred = rel.get('predicate') or rel.get('relation')
            target = rel.get('target')
            if pred and target:
                recurse(target, f"relations.{pred}")
                
    return items

def analyze_missed_connections():
    base_dir = Path(__file__).parent.parent
    
    # Files
    eval_results_file = base_dir / "Results" / "llm_eval_test_musique_v4_200_results.json"
    retrieval_results_file = base_dir / "Results" / "musique_retrieval_analysis_results.json"
    metadata_file = base_dir / "MuSiQue" / "musique_sample_200_metadata.json"
    
    print("Loading data...")
    with open(eval_results_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
        if isinstance(eval_data, dict) and 'results' in eval_data:
            eval_results = eval_data['results']
        else:
            eval_results = eval_data

    with open(retrieval_results_file, 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)
        # Map question text to retrieval result (since ID might be missing in eval)
        retrieval_map = {item['question']: item for item in retrieval_data}

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
        # Map ID -> Metadata Item
        meta_map = {item['id']: item for item in metadata_list}

    incorrect_questions = []
    
    # Filter for INCORRECT questions
    for res in eval_results:
        if res.get('evaluation', {}).get('verdict') == 'INCORRECT':
            incorrect_questions.append(res)
            
    print(f"Found {len(incorrect_questions)} INCORRECT questions.")
    
    connected_count = 0
    total_analyzed = 0
    
    print("\nAnalyzing shared metadata between Retrieved and Missed Supporting Facts...")
    
    for item in incorrect_questions:
        question_text = item['question']
        retrieval_item = retrieval_map.get(question_text)
        
        if not retrieval_item:
            continue
            
        q_id = retrieval_item['question_id']
        meta_item = meta_map.get(q_id)
        
        if not meta_item:
            continue
            
        gold_supporting = set(retrieval_item['supporting_facts'])
        retrieved_docs = set(retrieval_item['retrieved_docs'])
        
        # Identify Retrieved vs Missed Gold Docs
        supporting_retrieved = gold_supporting.intersection(retrieved_docs)
        supporting_missed = gold_supporting - retrieved_docs
        
        # We only care if there is at least one retrieved AND at least one missed
        if not supporting_retrieved or not supporting_missed:
            continue
            
        total_analyzed += 1
        
        # Build metadata map for context docs
        context_meta_map = {}
        for doc in meta_item.get('context_metadata', []):
            t = doc.get('title') or doc.get('metadata', {}).get('title')
            if t:
                context_meta_map[t] = doc.get('metadata', {})
        
        # Check for connections between any Retrieved Gold and any Missed Gold
        is_connected = False
        connections = []
        
        for ret_title in supporting_retrieved:
            if ret_title not in context_meta_map: continue
            
            ret_meta = set(flatten_metadata(context_meta_map[ret_title]))
            # Add title as value
            norm_ret_title = normalize_text(ret_title)
            if norm_ret_title:
                ret_meta.add(('self.title', norm_ret_title))
            
            ret_vals = {v for k, v in ret_meta}
            
            for miss_title in supporting_missed:
                if miss_title not in context_meta_map: continue
                
                miss_meta = set(flatten_metadata(context_meta_map[miss_title]))
                norm_miss_title = normalize_text(miss_title)
                if norm_miss_title:
                    miss_meta.add(('self.title', norm_miss_title))
                
                miss_vals = {v for k, v in miss_meta}
                
                shared = ret_vals.intersection(miss_vals)
                if shared:
                    is_connected = True
                    connections.append(f"{ret_title} <-> {miss_title} (Shared: {list(shared)[:3]})")
        
        if is_connected:
            connected_count += 1
            # print(f"\n[Q] {question_text}")
            # for c in connections:
            #     print(f"  - {c}")

    if total_analyzed > 0:
        ratio = (connected_count / total_analyzed) * 100
        print(f"\nAnalysis Result:")
        print(f"Total INCORRECT questions with mixed retrieval (some found, some missed): {total_analyzed}")
        print(f"Questions where Retrieved and Missed docs share metadata: {connected_count}")
        print(f"Ratio: {ratio:.2f}%")
    else:
        print("\nNo suitable questions found for analysis (e.g., either all found or all missed).")

if __name__ == "__main__":
    analyze_missed_connections()
