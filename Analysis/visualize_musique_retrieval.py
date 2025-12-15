import json
import re
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

def escape_xml(s):
    if not isinstance(s, str):
        s = str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;").replace("'", "&apos;")

def visualize_musique_retrieval():
    base_dir = Path(__file__).parent.parent
    
    # Files
    results_file = base_dir / "Results" / "musique_retrieval_analysis_results.json"
    original_file = base_dir / "MuSiQue" / "musique_sample_200.json"
    metadata_file = base_dir / "MuSiQue" / "musique_sample_200_metadata.json"
    output_graph = base_dir / "musique_retrieval_graph.graphml"
    
    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        print("Please run 'run_musique_retrieval.py' first.")
        return

    print("Loading data...")
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
        
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
        # Map ID -> Item
        original_map = {item['_id']: item for item in original_data}
        
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
        # Map ID -> Metadata Item
        meta_map = {item['id']: item for item in metadata_list}

    print("Generating GraphML...")
    f_graph = open(output_graph, 'w', encoding='utf-8')
    f_graph.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f_graph.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"  \n')
    f_graph.write('    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
    f_graph.write('    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n')
    f_graph.write('     http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n')
    
    # Node Keys
    f_graph.write('  <key id="d0" for="node" attr.name="type" attr.type="string"/>\n') 
    # types: question, supporting_retrieved, supporting_missed, distractor_retrieved
    f_graph.write('  <key id="d1" for="node" attr.name="label" attr.type="string"/>\n')
    
    # Edge Keys
    f_graph.write('  <key id="d2" for="edge" attr.name="relation" attr.type="string"/>\n')
    f_graph.write('  <key id="d3" for="edge" attr.name="shared_info" attr.type="string"/>\n')
    
    f_graph.write('  <graph id="G" edgedefault="undirected">\n')
    
    edge_counter = 0
    
    for res in results_data:
        q_id = res['question_id']
        question_text = res['question']
        
        # Get Original Context & Metadata
        orig_item = original_map.get(q_id)
        meta_item = meta_map.get(q_id)
        
        if not orig_item or not meta_item:
            continue
            
        # 1. Identify Documents
        gold_supporting = set(res['supporting_facts'])
        retrieved_docs = set(res['retrieved_docs'])
        
        # Filter retrieved docs: Must be in original context
        # MuSiQue structure: 'paragraphs' list of dicts with 'title', 'paragraph_text', ...
        # Or 'context' list? Let's check structure. 
        # HotpotQA has 'context' as list of [title, sentences].
        # MuSiQue sample usually follows similar structure or has 'paragraphs'.
        # Let's assume 'paragraphs' based on standard MuSiQue, but check 'context' if present.
        
        valid_titles = set()
        if 'paragraphs' in orig_item:
            for p in orig_item['paragraphs']:
                valid_titles.add(p['title'])
        elif 'context' in orig_item: # HotpotQA style
             for p in orig_item['context']:
                # context is [title, text] list
                valid_titles.add(p[0])
        
        # Filter retrieved
        filtered_retrieved = {t for t in retrieved_docs if t in valid_titles}
        
        # Union of docs to visualize: Gold + Filtered Retrieved
        docs_to_visualize = gold_supporting.union(filtered_retrieved)
        
        if not docs_to_visualize:
            continue

        # Question Node
        q_node_id = f"Q_{q_id}"
        f_graph.write(f'    <node id="{q_node_id}">\n')
        f_graph.write(f'      <data key="d0">question</data>\n')
        f_graph.write(f'      <data key="d1">{escape_xml(question_text)}</data>\n')
        f_graph.write(f'    </node>\n')
        
        # Document Nodes & Metadata
        doc_flat_meta = {}
        
        # Build metadata map for context docs
        context_meta_map = {}
        for doc in meta_item.get('context_metadata', []):
            t = doc.get('title') or doc.get('metadata', {}).get('title')
            if t:
                context_meta_map[t] = doc.get('metadata', {})
        
        for title in docs_to_visualize:
            # Determine Type
            is_gold = title in gold_supporting
            is_retrieved = title in filtered_retrieved
            
            if is_gold and is_retrieved:
                node_type = "supporting_retrieved"
            elif is_gold and not is_retrieved:
                node_type = "supporting_missed"
            elif not is_gold and is_retrieved:
                node_type = "distractor_retrieved"
            else:
                continue # Should not happen based on set union
            
            doc_node_id = f"D_{q_id}_{escape_xml(title)}"
            f_graph.write(f'    <node id="{doc_node_id}">\n')
            f_graph.write(f'      <data key="d0">{node_type}</data>\n')
            f_graph.write(f'      <data key="d1">{escape_xml(title)}</data>\n')
            f_graph.write(f'    </node>\n')
            
            # Edge: Question -> Doc
            f_graph.write(f'    <edge id="e{edge_counter}" source="{q_node_id}" target="{doc_node_id}">\n')
            f_graph.write(f'      <data key="d2">has_context</data>\n')
            f_graph.write(f'    </edge>\n')
            edge_counter += 1
            
            # Prepare metadata for linking
            if title in context_meta_map:
                flat = set(flatten_metadata(context_meta_map[title]))
                norm_title = normalize_text(title)
                if norm_title:
                    flat.add(('self.title', norm_title))
                doc_flat_meta[title] = flat
            else:
                doc_flat_meta[title] = set()

        # Metadata Links
        titles = list(docs_to_visualize)
        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                t1 = titles[i]
                t2 = titles[j]
                
                meta1 = doc_flat_meta.get(t1, set())
                meta2 = doc_flat_meta.get(t2, set())
                
                vals1 = {v for k, v in meta1}
                vals2 = {v for k, v in meta2}
                
                shared_vals = vals1.intersection(vals2)
                
                if shared_vals:
                    shared_info = list(shared_vals)[:5]
                    info_str = ", ".join(shared_info)
                    
                    n1_id = f"D_{q_id}_{escape_xml(t1)}"
                    n2_id = f"D_{q_id}_{escape_xml(t2)}"
                    
                    f_graph.write(f'    <edge id="e{edge_counter}" source="{n1_id}" target="{n2_id}">\n')
                    f_graph.write(f'      <data key="d2">shares_metadata</data>\n')
                    f_graph.write(f'      <data key="d3">{escape_xml(info_str)}</data>\n')
                    f_graph.write(f'    </edge>\n')
                    edge_counter += 1

    f_graph.write('  </graph>\n')
    f_graph.write('</graphml>\n')
    f_graph.close()
    print(f"GraphML saved to {output_graph}")

if __name__ == "__main__":
    visualize_musique_retrieval()
