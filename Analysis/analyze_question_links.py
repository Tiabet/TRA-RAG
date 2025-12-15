import json
import re
from collections import defaultdict
from pathlib import Path
import os

def normalize_text(text):
    if text is None:
        return ""
    s = str(text).lower()
    s = s.replace(',', '')
    return s.strip()

def flatten_metadata(meta):
    """
    Recursively extracts (key_path, value) pairs from metadata.
    Handles nested dictionaries and lists.
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

    # Handle Attributes
    # We start with 'attributes' as the root path for attributes
    recurse(meta.get('attributes', {}), 'attributes')
    
    # Handle Relations
    # Relations are a list of dicts, usually with 'relation'/'predicate' and 'target'.
    # We want to map them to relations.<predicate> = <target_value>
    for rel in meta.get('relations', []):
        if isinstance(rel, dict):
            pred = rel.get('predicate') or rel.get('relation')
            target = rel.get('target')
            
            if pred and target:
                # Recurse on target, using relations.<pred> as the path
                recurse(target, f"relations.{pred}")
                
    return items

def escape_xml(s):
    if not isinstance(s, str):
        s = str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;").replace("'", "&apos;")

def analyze_question_links(metadata_file_path, qa_file_path):
    print(f"Loading Metadata from {metadata_file_path}...")
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    print(f"Loading QA Data from {qa_file_path}...")
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    # Output structures
    json_output = []
    
    # GraphML Setup
    graphml_path = Path("question_metadata_graph.graphml")
    f_graph = open(graphml_path, 'w', encoding='utf-8')
    f_graph.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f_graph.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"  \n')
    f_graph.write('    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
    f_graph.write('    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n')
    f_graph.write('     http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n')
    # Node Keys
    f_graph.write('  <key id="d0" for="node" attr.name="type" attr.type="string"/>\n') # question, supporting_doc, distractor_doc
    f_graph.write('  <key id="d1" for="node" attr.name="label" attr.type="string"/>\n')
    # Edge Keys
    f_graph.write('  <key id="d2" for="edge" attr.name="relation" attr.type="string"/>\n') # has_context, shares_metadata
    f_graph.write('  <key id="d3" for="edge" attr.name="shared_info" attr.type="string"/>\n') # key=value info
    
    f_graph.write('  <graph id="G" edgedefault="undirected">\n')

    edge_counter = 0
    
    # Iterate through questions
    # Metadata file uses 'id', QA file uses '_id'
    meta_map = {item['id']: item for item in metadata_list}

    for qa_item in qa_data:
        q_id = qa_item['_id']
        question_text = qa_item['question']
        
        # Get corresponding metadata item
        meta_item = meta_map.get(q_id)
        if not meta_item:
            continue

        # Identify Supporting Facts (Titles)
        supporting_titles = set()
        for fact in qa_item.get('supporting_facts', []):
            if fact:
                supporting_titles.add(fact[0])

        # Identify Context Documents and their Metadata
        context_docs = {} # Title -> Metadata Dict
        
        for doc in meta_item.get('context_metadata', []):
            title = doc.get('title') or doc.get('metadata', {}).get('title')
            if title:
                context_docs[title] = doc.get('metadata', {})

        # Prepare JSON entry
        q_analysis = {
            "question_id": q_id,
            "question": question_text,
            "supporting_facts": list(supporting_titles),
            "shared_metadata_among_supporting": [],
            "shared_metadata_all_context": [] 
        }

        # --- Graph Generation: Question Node ---
        q_node_id = f"Q_{q_id}"
        f_graph.write(f'    <node id="{q_node_id}">\n')
        f_graph.write(f'      <data key="d0">question</data>\n')
        f_graph.write(f'      <data key="d1">{escape_xml(question_text)}</data>\n')
        f_graph.write(f'    </node>\n')

        # --- Analyze Metadata Sharing ---
        doc_flat_meta = {} # Title -> Set of (key, value) tuples
        
        for title, meta in context_docs.items():
            flat = set(flatten_metadata(meta))
            # Add title as a value so other docs referring to this one link to it
            norm_title = normalize_text(title)
            if norm_title:
                flat.add(('self.title', norm_title))
            
            doc_flat_meta[title] = flat
            
            # --- Graph Generation: Document Node ---
            doc_node_id = f"D_{q_id}_{escape_xml(title)}" 
            
            doc_type = "supporting_doc" if title in supporting_titles else "distractor_doc"
            
            f_graph.write(f'    <node id="{doc_node_id}">\n')
            f_graph.write(f'      <data key="d0">{doc_type}</data>\n')
            f_graph.write(f'      <data key="d1">{escape_xml(title)}</data>\n')
            f_graph.write(f'    </node>\n')
            
            # Edge: Question -> Doc
            f_graph.write(f'    <edge id="e{edge_counter}" source="{q_node_id}" target="{doc_node_id}">\n')
            f_graph.write(f'      <data key="d2">has_context</data>\n')
            f_graph.write(f'    </edge>\n')
            edge_counter += 1

        # Compare Supporting Docs specifically for JSON
        supp_list = list(supporting_titles)
        valid_supp_list = [t for t in supp_list if t in doc_flat_meta]
        
        if len(valid_supp_list) > 1:
            val_map = defaultdict(list)
            for title in valid_supp_list:
                for k, v in doc_flat_meta[title]:
                    val_map[v].append((title, k))
            
            for val, occurrences in val_map.items():
                if len(occurrences) > 1:
                    docs_involved = set(x[0] for x in occurrences)
                    keys_involved = set(x[1] for x in occurrences)
                    
                    if len(docs_involved) > 1:
                        q_analysis["shared_metadata_among_supporting"].append({
                            "value": val,
                            "keys": list(keys_involved),
                            "docs": list(docs_involved)
                        })

        json_output.append(q_analysis)

        # --- Graph Generation: Metadata Edges (All Context) ---
        titles = list(context_docs.keys())
        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                t1 = titles[i]
                t2 = titles[j]
                
                meta1 = doc_flat_meta[t1]
                meta2 = doc_flat_meta[t2]
                
                vals1 = {v for k, v in meta1}
                vals2 = {v for k, v in meta2}
                
                shared_vals = vals1.intersection(vals2)
                
                if shared_vals:
                    shared_info = []
                    for val in list(shared_vals)[:5]: 
                        shared_info.append(f"{val}")
                    
                    # Add to JSON (All Context)
                    q_analysis["shared_metadata_all_context"].append({
                        "doc1": t1,
                        "doc2": t2,
                        "shared_values": list(shared_vals)
                    })

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
    print(f"GraphML saved to {graphml_path.absolute()}")
    
    json_path = Path("question_supporting_links.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"JSON analysis saved to {json_path.absolute()}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    metadata_file = base_dir / "HotpotQA" / "hotpotqa_sample_200_metadata.json"
    qa_file = base_dir / "HotpotQA" / "hotpotqa_sample_200.json"
    
    analyze_question_links(metadata_file, qa_file)
