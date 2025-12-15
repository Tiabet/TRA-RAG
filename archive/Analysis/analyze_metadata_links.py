import json
import re
from collections import defaultdict
from pathlib import Path
import statistics
import os

def normalize_text(text):
    """
    Simple preprocessing:
    1. Convert to string
    2. Lowercase
    3. Remove commas
    4. Strip whitespace
    """
    if text is None:
        return ""
    s = str(text).lower()
    s = s.replace(',', '')
    return s.strip()

def extract_leaf_values(obj, values_set):
    """Recursively extract leaf values from a dictionary or list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            extract_leaf_values(v, values_set)
    elif isinstance(obj, list):
        for item in obj:
            extract_leaf_values(item, values_set)
    elif isinstance(obj, (str, int, float, bool)):
        val = normalize_text(obj)
        if val: # Ignore empty strings
            values_set.add(val)

def escape_xml(s):
    if not isinstance(s, str):
        s = str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;").replace("'", "&apos;")

def analyze_links(metadata_file_path, qa_file_path):
    print(f"Loading Metadata from {metadata_file_path}...")
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        metadata_data = json.load(f)

    print(f"Loading QA Data from {qa_file_path}...")
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    # 1. Collect all Supporting Fact Titles
    supporting_fact_titles = set()
    for item in qa_data:
        for fact in item.get('supporting_facts', []):
            # fact is [title, sent_id]
            if fact and len(fact) > 0:
                supporting_fact_titles.add(fact[0])
    
    print(f"Total Unique Supporting Fact Documents found: {len(supporting_fact_titles)}")

    # 2. Build Graph Connections
    # Map: Value -> Set of Titles (Passages)
    value_to_titles = defaultdict(set)
    
    # Track unique documents by title
    processed_titles = set()
    
    # Store document metadata for graph attributes later
    doc_nodes = {} 

    doc_count = 0
    
    for qa_item in metadata_data:
        context_metadata = qa_item.get('context_metadata', [])
        
        for doc in context_metadata:
            meta = doc.get('metadata', {})
            if not meta:
                continue
                
            title = meta.get('title')
            if not title:
                title = doc.get('title') # Fallback
            
            if not title:
                continue
                
            if title in processed_titles:
                continue
            
            processed_titles.add(title)
            doc_nodes[title] = {'type': 'document'}
            doc_count += 1
            
            # Extract values for this document
            doc_values = set()
            
            # 1. Title itself is a value
            doc_values.add(normalize_text(title))
            
            # 2. Attributes
            attributes = meta.get('attributes', {})
            extract_leaf_values(attributes, doc_values)
            
            # 3. Relations (targets)
            relations = meta.get('relations', [])
            for rel in relations:
                if isinstance(rel, dict):
                    target = rel.get('target')
                    if target:
                        doc_values.add(normalize_text(target))
            
            # Register values
            for val in doc_values:
                value_to_titles[val].add(title)

    # Filter values that create links (appear in > 1 document)
    linking_values = {k: v for k, v in value_to_titles.items() if len(v) > 1}
    
    # Build Adjacency List for Documents
    # Doc -> Set of connected Docs
    doc_adjacency = defaultdict(set)
    
    for val, titles in linking_values.items():
        titles_list = list(titles)
        for i in range(len(titles_list)):
            for j in range(i + 1, len(titles_list)):
                doc_a = titles_list[i]
                doc_b = titles_list[j]
                doc_adjacency[doc_a].add(doc_b)
                doc_adjacency[doc_b].add(doc_a)

    # 3. Analyze Isolated Nodes
    isolated_nodes = []
    linked_nodes = []
    
    for title in processed_titles:
        if title not in doc_adjacency or len(doc_adjacency[title]) == 0:
            isolated_nodes.append(title)
        else:
            linked_nodes.append(title)
            
    print(f"\n--- Link Analysis ---")
    print(f"Total Documents: {len(processed_titles)}")
    print(f"Linked Documents: {len(linked_nodes)} ({len(linked_nodes)/len(processed_titles)*100:.2f}%)")
    print(f"Isolated Documents: {len(isolated_nodes)} ({len(isolated_nodes)/len(processed_titles)*100:.2f}%)")
    
    # 4. Check Isolated Nodes against Supporting Facts
    isolated_supporting_count = 0
    for title in isolated_nodes:
        if title in supporting_fact_titles:
            isolated_supporting_count += 1
            
    if len(isolated_nodes) > 0:
        ratio = isolated_supporting_count / len(isolated_nodes)
        print(f"\n--- Risk Analysis ---")
        print(f"Isolated Documents that are Supporting Facts: {isolated_supporting_count}")
        print(f"Percentage of Isolated Docs that are Supporting Facts: {ratio*100:.2f}%")
    else:
        print("\nNo isolated nodes found.")

    # 5. Generate GraphML
    print(f"\nGenerating GraphML...")
    
    graphml_path = Path("metadata_graph.graphml")
    
    with open(graphml_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"  \n')
        f.write('    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
        f.write('    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n')
        f.write('     http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n')
        f.write('  <key id="d0" for="node" attr.name="type" attr.type="string"/>\n')
        f.write('  <key id="d1" for="node" attr.name="is_supporting" attr.type="boolean"/>\n')
        f.write('  <key id="d2" for="edge" attr.name="shared_value" attr.type="string"/>\n')
        f.write('  <graph id="G" edgedefault="undirected">\n')
        
        # Write Nodes
        for title in processed_titles:
            is_supp = "true" if title in supporting_fact_titles else "false"
            f.write(f'    <node id="{escape_xml(title)}">\n')
            f.write(f'      <data key="d0">document</data>\n')
            f.write(f'      <data key="d1">{is_supp}</data>\n')
            f.write(f'    </node>\n')
            
        # Write Edges
        edge_id = 0
        
        for val, titles in linking_values.items():
            titles_list = list(titles)
            for i in range(len(titles_list)):
                for j in range(i + 1, len(titles_list)):
                    doc_a = titles_list[i]
                    doc_b = titles_list[j]
                    
                    f.write(f'    <edge id="e{edge_id}" source="{escape_xml(doc_a)}" target="{escape_xml(doc_b)}">\n')
                    f.write(f'      <data key="d2">{escape_xml(val)}</data>\n')
                    f.write(f'    </edge>\n')
                    edge_id += 1

        f.write('  </graph>\n')
        f.write('</graphml>\n')
        
    print(f"GraphML saved to {graphml_path.absolute()}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    metadata_file = base_dir / "HotpotQA" / "hotpotqa_sample_200_metadata.json"
    qa_file = base_dir / "HotpotQA" / "hotpotqa_sample_200.json"
    
    analyze_links(metadata_file, qa_file)
