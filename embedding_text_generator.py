#!/usr/bin/env python3
"""
Embedding Text Generator
========================
Converts hierarchical metadata into embedding-ready natural language texts.

Structure: title + key path + value
Format: "{title}의 {key_path}는 {value}이다" (Korean)
    or: "The {key_path} of {title} is {value}" (English)

Includes: ALL values (strings, numbers, lists, nested objects)
"""

import json
import sqlite3
from typing import List, Dict, Any, Tuple
from pathlib import Path


class EmbeddingTextGenerator:
    """Generate embedding-ready texts from metadata."""
    
    def __init__(self, language: str = "ko"):
        """
        Args:
            language: "ko" for Korean, "en" for English
        """
        self.language = language
    
    def extract_embedding_texts(self, title: str, metadata: Dict) -> List[Dict]:
        """
        Extract all embedding texts from metadata.
        
        Args:
            title: Entity title (e.g., "The Wolf of Wall Street (2013 film)")
            metadata: Full metadata dict
            
        Returns:
            List of dicts with:
                - text: Natural language text for embedding
                - key_path: Hierarchical key path (e.g., "director.name")
                - value: Original value
        """
        results = []
        
        # Process the metadata, skipping the wrapper
        inner_metadata = metadata.get('metadata', metadata)
        
        # Extract from attributes
        if 'attributes' in inner_metadata:
            for key, value in inner_metadata['attributes'].items():
                # Clean value
                cleaned_value = self._clean_value(value)

                # If this is a list of dicts (e.g., rivers, bridges), split into per-item paths
                if self._is_list_of_dicts(cleaned_value):
                    self._add_list_of_dict_items(title, [key], cleaned_value, results, is_relation=False)
                else:
                    # Add result (Attribute type)
                    self._add_result(title, [key], cleaned_value, results, is_relation=False)
        
        # Extract from relations
        if 'relations' in inner_metadata:
            for relation in inner_metadata.get('relations', []):
                # Relation is usually a dict: { "relation": "...", "target": ... }
                if not isinstance(relation, dict):
                    continue
                
                relation_type = relation.get('relation', 'related_to')

                # For relations only: if target is a list, emit one fact per target element.
                target = relation.get('target', None)
                if isinstance(target, list):
                    for t in target:
                        value_dict = relation.copy()
                        value_dict.pop('relation', None)
                        value_dict['target'] = t
                        cleaned_value = self._clean_value(value_dict)
                        self._add_result(title, [relation_type], cleaned_value, results, is_relation=True)
                else:
                    # Default behavior (single target / dict target)
                    value_dict = relation.copy()
                    value_dict.pop('relation', None)
                    cleaned_value = self._clean_value(value_dict)
                    self._add_result(title, [relation_type], cleaned_value, results, is_relation=True)
        
        # Extract top-level fields (excluding wrapper keys)
        skip_keys = {'title', 'attributes', 'relations', 'metadata'}
        for key, value in inner_metadata.items():
            if key in skip_keys:
                continue
            
            cleaned_value = self._clean_value(value)
            print(f"DEBUG: Key={key}, Type={type(cleaned_value)}")

            if self._is_list_of_dicts(cleaned_value):
                self._add_list_of_dict_items(title, [key], cleaned_value, results, is_relation=False)
            else:
                self._add_result(title, [key], cleaned_value, results, is_relation=False)
            
            # [Hybrid Approach] Also add flattened leaf nodes for complex objects
            # This ensures specific details are not lost in the grouped text
            if isinstance(cleaned_value, dict):
                print(f"DEBUG: Calling flatten for {key}")
                self._add_flattened_results(title, [key], cleaned_value, results)
        
        return results

    def _add_flattened_results(
        self,
        title: str,
        current_path: List[str],
        value: Any,
        results: List[Dict]
    ):
        """Recursively add flattened leaf nodes."""
        print(f"DEBUG: Flattening {current_path} -> {type(value)}")
        if isinstance(value, dict):
            for k, v in value.items():
                self._add_flattened_results(title, current_path + [k], v, results)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                seg = None
                if isinstance(item, dict):
                    seg = self._pick_item_label(item)
                seg = self._sanitize_path_segment(seg) if seg else f"[{i}]"
                self._add_flattened_results(title, current_path + [seg], item, results)
        else:
            # Leaf node
            self._add_result(title, current_path, value, results, is_relation=False)

    def _is_list_of_dicts(self, value: Any) -> bool:
        return isinstance(value, list) and len(value) > 0 and all(isinstance(x, dict) for x in value)

    def _pick_item_label(self, item: Dict[str, Any]) -> str:
        """Pick a stable, human-readable identifier for list-of-dict items."""
        for k in ('name', 'title', 'id'):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def _sanitize_path_segment(self, seg: str) -> str:
        seg = seg.replace('.', '_')
        seg = seg.replace('\n', ' ').replace('\r', ' ').strip()
        # Keep it reasonably short to avoid huge key_path strings
        if len(seg) > 80:
            seg = seg[:80]
        return seg

    def _add_list_of_dict_items(
        self,
        title: str,
        base_path: List[str],
        items: List[Dict[str, Any]],
        results: List[Dict],
        is_relation: bool = False
    ):
        """Add one embedding entry per dict item in a list.

        Example:
          rivers -> rivers.River Test (value is the whole river dict)
        """
        for i, item in enumerate(items):
            label = self._pick_item_label(item)
            seg = self._sanitize_path_segment(label) if label else f"[{i}]"
            # Store the whole object as value (keeps nested info like bridges together)
            self._add_result(title, base_path + [seg], item, results, is_relation=is_relation)

    def _clean_value(self, value: Any) -> Any:
        """
        Recursively normalize values.
        """
        if isinstance(value, dict):
            return {k: self._clean_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._clean_value(item) for item in value]
        else:
            return value
    
    def _format_value_natural(self, value: Any) -> str:
        """
        Format value into natural language string.
        
        - Dict: "key is value, key is value"
        - List: "item, item"
        - String: "value"
        """
        if isinstance(value, dict):
            parts = []
            for k, v in value.items():
                # Clean key: replace underscores with spaces
                clean_k = k.replace('_', ' ')
                # Recursively format value
                clean_v = self._format_value_natural(v)
                parts.append(f"{clean_k} is {clean_v}")
            return ", ".join(parts)
        
        elif isinstance(value, list):
            return ", ".join([self._format_value_natural(item) for item in value])
        
        else:
            return str(value)

    def _add_result(
        self,
        title: str,
        key_path: List[str],
        value: Any,
        results: List[Dict],
        is_relation: bool = False
    ):
        """Add a result to the accumulator."""
        if value is None:
            return
        
        # Convert value to string (JSON for complex objects) - kept for 'value' field
        if isinstance(value, (dict, list)):
            value_json = json.dumps(value, ensure_ascii=False)
        else:
            value_json = str(value)
        
        # Skip empty values
        if not value_json.strip():
            return
        
        # Build key path string
        key_path_str = ".".join(key_path)
        
        # Generate natural language text
        if self.language == "ko":
            text = self._format_korean(title, key_path, value_json)
        else:
            text = self._format_english(title, key_path, value, is_relation)
        
        results.append({
            'text': text,
            'key_path': key_path_str,
            'value': value_json,
            'title': title
        })
    
    def _format_korean(self, title: str, key_path: List[str], value: str) -> str:
        """Format as Korean natural language."""
        if len(key_path) == 1:
            return f"{title}의 {key_path[0]}은/는 {value}이다"
        else:
            path_str = "의 ".join(key_path)
            return f"{title}의 {path_str}은/는 {value}이다"
    
    def _format_english(self, title: str, key_path: List[str], value: Any, is_relation: bool) -> str:
        """
        Format as English natural language.
        
        Attributes: "The {key} of {title} is {formatted_value}"
        Relations: "{title} is {relation} {formatted_value}"
        """
        # Clean key path (replace underscores with spaces)
        clean_keys = [k.replace('_', ' ') for k in key_path]
        
        if is_relation:
            # Relation format: Title is Relation Value
            relation_name = clean_keys[0]
            
            # Handle value for relation
            if isinstance(value, dict) and 'target' in value:
                # If target exists, use it as the primary value
                # We ignore other keys in the relation dict as per instruction
                val_str = self._format_value_natural(value['target'])
            else:
                # Fallback: format the whole value naturally
                val_str = self._format_value_natural(value)
            
            return f"{title} is {relation_name} {val_str}"
            
        else:
            # Attribute format: The Key of Title is Value
            
            # Format value naturally
            val_str = self._format_value_natural(value)
            
            if len(clean_keys) == 1:
                return f"The {clean_keys[0]} of {title} is {val_str}"
            else:
                path_str = " of the ".join(reversed(clean_keys))
                return f"The {path_str} of {title} is {val_str}"


def generate_embedding_texts_from_db(
    db_path: str = 'HotpotQA/metadata_v3.db',
    output_path: str = 'HotpotQA/embedding_texts.json',
    language: str = "ko"
) -> List[Dict]:
    """
    Generate embedding texts from metadata database.
    
    Args:
        db_path: Path to metadata_v3.db
        output_path: Path to save embedding texts JSON
        language: "ko" or "en"
        
    Returns:
        List of all embedding text entries
    """
    print("="*80)
    print("Generating Embedding Texts from Metadata DB")
    print("="*80)
    
    generator = EmbeddingTextGenerator(language=language)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all metadata entries
    # Support both the new schema (doc_id/source_title/entity_title) and legacy (title).
    try:
        cursor.execute("SELECT doc_id, source_title, entity_title, metadata_json FROM metadata")
        rows = cursor.fetchall()
        schema_mode = "v2"
    except Exception:
        cursor.execute("SELECT title, metadata_json FROM metadata")
        rows = cursor.fetchall()
        schema_mode = "legacy"
    
    print(f"\nProcessing {len(rows)} metadata entries...")
    
    all_texts = []
    
    for idx, row in enumerate(rows):
        if schema_mode == "v2":
            doc_id = row['doc_id']
            source_title = row['source_title']
            entity_title = row['entity_title']
            metadata = json.loads(row['metadata_json'])
        else:
            doc_id = None
            source_title = row['title']
            entity_title = row['title']
            metadata = json.loads(row['metadata_json'])
        
        # Extract embedding texts
        # IMPORTANT:
        # - Use entity_title (metadata title) in the embedding text.
        # - Keep source_title (outer title) for later passage lookup + evaluation.
        texts = generator.extract_embedding_texts(entity_title, metadata)
        for t in texts:
            if doc_id is not None:
                t['doc_id'] = doc_id
            t['source_title'] = source_title
            t['entity_title'] = entity_title
        all_texts.extend(texts)
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(rows)} entries...")
    
    conn.close()
    
    print(f"\n[OK] Generated {len(all_texts)} embedding texts")
    
    # Save to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Saved to {output_path}")
    return all_texts

def generate_embedding_texts_from_json(
    json_path: str = 'HotpotQA/hotpotqa_sample_200_metadata.json',
    output_path: str = 'HotpotQA/embedding_texts.json',
    language: str = "ko"
) -> List[Dict]:
    """
    Generate embedding texts from metadata JSON file (list of QA pairs with context_metadata).
    """
    print("="*80)
    print("Generating Embedding Texts from Metadata JSON")
    print("="*80)
    
    generator = EmbeddingTextGenerator(language=language)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nProcessing {len(data)} QA pairs...")
    
    all_texts = []
    
    for idx, item in enumerate(data):
        qid = item.get('_id') or item.get('id') or str(idx)
        context_metadata = item.get('context_metadata', [])
        for ci, meta_entry in enumerate(context_metadata):
            source_title = meta_entry.get('title')
            if not source_title:
                continue

            metadata = meta_entry.get('metadata', meta_entry)
            entity_title = (metadata or {}).get('title') or source_title
            # Prefer doc_id provided by build_metadata (may be corpus_idx).
            doc_id = meta_entry.get('doc_id')
            if not doc_id:
                ctx_idx = meta_entry.get('ctx_idx')
                if ctx_idx is not None:
                    doc_id = f"{qid}::ctx{int(ctx_idx)}"
                else:
                    doc_id = f"{qid}::ctx{ci}"
            
            # Extract embedding texts
            texts = generator.extract_embedding_texts(entity_title, metadata)
            for t in texts:
                t['doc_id'] = doc_id
                t['source_title'] = source_title
                t['entity_title'] = entity_title
            all_texts.extend(texts)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(data)} QA pairs...")
    
    unique_doc_ids = len(set(t.get('doc_id') for t in all_texts if t.get('doc_id')))
    unique_entity_titles = len(set(t.get('entity_title') for t in all_texts if t.get('entity_title')))
    print(f"\n[OK] Generated {len(all_texts)} embedding texts from {unique_doc_ids} docs ({unique_entity_titles} entity titles)")
    
    # Save to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Saved to {output_path}")
    return all_texts

def test_single_metadata():
    """Test with a single metadata entry."""
    
    # Sample metadata (The Wolf of Wall Street)
    sample = {
        "title": "The Wolf of Wall Street (2013 film)",
        "type": "WorkOfArt",
        "subtype": "Film",
        "attributes": {
            "release_year": 2013,
            "nationality": "American",
            "genre": ["biographical", "black comedy", "crime"],
            "director": {
                "name": "Martin Scorsese",
                "type": "Person",
                "subtype": "Director"
            },
            "cast": [
                {
                    "name": "Leonardo DiCaprio",
                    "role": "Jordan Belfort",
                    "is_producer": True,
                    "type": "Person",
                    "subtype": "Actor"
                },
                {
                    "name": "Jonah Hill",
                    "role": "Donnie Azoff",
                    "type": "Person",
                    "subtype": "Actor"
                }
            ],
            "plot_summary": "It recounts Belfort's perspective on his career as a stockbroker..."
        },
        "relations": [
            {
                "relation": "based_on",
                "target": {
                    "title": "The Wolf of Wall Street",
                    "type": "WorkOfArt",
                    "subtype": "Book"
                },
                "author": {
                    "name": "Jordan Belfort",
                    "type": "Person"
                }
            }
        ]
    }
    
    print("="*80)
    print("Test: Single Metadata Entry")
    print("="*80)
    
    print("\n[Input Metadata]")
    print(json.dumps(sample, indent=2, ensure_ascii=False)[:500] + "...")
    
    # Test Korean
    print("\n" + "-"*40)
    print("[Korean Output]")
    print("-"*40)
    
    generator_ko = EmbeddingTextGenerator(language="ko")
    texts_ko = generator_ko.extract_embedding_texts(sample['title'], sample)
    
    for i, entry in enumerate(texts_ko, 1):
        print(f"{i:2d}. {entry['text']}")
        print(f"    key_path: {entry['key_path']}, value: {entry['value']}")
    
    # Test English
    print("\n" + "-"*40)
    print("[English Output]")
    print("-"*40)
    
    generator_en = EmbeddingTextGenerator(language="en")
    texts_en = generator_en.extract_embedding_texts(sample['title'], sample)
    
    for i, entry in enumerate(texts_en, 1):
        print(f"{i:2d}. {entry['text']}")
    
    print(f"\n[OK] Total texts generated: {len(texts_ko)}")


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_single_metadata()
    else:
        # Check for JSON first
        json_path = 'HotpotQA/hotpotqa_sample_200_metadata.json'
        if os.path.exists(json_path):
            generate_embedding_texts_from_json(
                json_path=json_path,
                output_path='HotpotQA/embedding_texts.json',
                language="en"
            )
        else:
            # Fallback to DB
            generate_embedding_texts_from_db(
                db_path='HotpotQA/metadata_v3.db',
                output_path='HotpotQA/embedding_texts.json',
                language="en"
            )
