#!/usr/bin/env python3
"""
Embedding Text Generator
========================
Converts hierarchical metadata into embedding-ready natural language texts.

Structure: title + key path + value
Format: "{title}의 {key_path}는 {value}이다" (Korean)
    or: "The {key_path} of {title} is {value}" (English)

Ignores: type, subtype (meaningless for embedding)
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
                # Skip type/subtype
                if key in ('type', 'subtype'):
                    continue
                
                # Clean value (remove type/subtype from nested dicts)
                cleaned_value = self._clean_value(value)
                
                # Add result (Attribute type)
                self._add_result(title, [key], cleaned_value, results, is_relation=False)
        
        # Extract from relations
        if 'relations' in inner_metadata:
            for relation in inner_metadata.get('relations', []):
                # Relation is usually a dict: { "relation": "...", "target": ... }
                if not isinstance(relation, dict):
                    continue
                
                relation_type = relation.get('relation', 'related_to')
                
                # Clean the whole relation object, but exclude 'relation' key from the value
                value_dict = relation.copy()
                if 'relation' in value_dict:
                    del value_dict['relation']
                
                cleaned_value = self._clean_value(value_dict)
                
                # Add result (Relation type)
                self._add_result(title, [relation_type], cleaned_value, results, is_relation=True)
        
        # Extract top-level fields (excluding type, subtype, title, attributes, relations, metadata)
        skip_keys = {'type', 'subtype', 'title', 'attributes', 'relations', 'metadata'}
        for key, value in inner_metadata.items():
            if key in skip_keys:
                continue
            
            cleaned_value = self._clean_value(value)
            print(f"DEBUG: Key={key}, Type={type(cleaned_value)}")
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
            for item in value:
                self._add_flattened_results(title, current_path, item, results)
        else:
            # Leaf node
            self._add_result(title, current_path, value, results, is_relation=False)

    def _clean_value(self, value: Any) -> Any:
        """
        Recursively remove type and subtype from dictionary values.
        """
        if isinstance(value, dict):
            return {
                k: self._clean_value(v)
                for k, v in value.items()
                if k not in ('type', 'subtype')
            }
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
    cursor.execute("SELECT title, metadata_json FROM metadata")
    rows = cursor.fetchall()
    
    print(f"\nProcessing {len(rows)} metadata entries...")
    
    all_texts = []
    
    for idx, row in enumerate(rows):
        title = row['title']
        metadata = json.loads(row['metadata_json'])
        
        # Extract embedding texts
        texts = generator.extract_embedding_texts(title, metadata)
        all_texts.extend(texts)
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(rows)} entries...")
    
    conn.close()
    
    print(f"\n✓ Generated {len(all_texts)} embedding texts")
    
    # Save to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved to {output_path}")
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
    processed_titles = set()
    
    for idx, item in enumerate(data):
        context_metadata = item.get('context_metadata', [])
        for meta_entry in context_metadata:
            title = meta_entry.get('title')
            if not title or title in processed_titles:
                continue
            
            processed_titles.add(title)
            metadata = meta_entry.get('metadata', meta_entry)
            
            # Extract embedding texts
            texts = generator.extract_embedding_texts(title, metadata)
            all_texts.extend(texts)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(data)} QA pairs...")
    
    print(f"\n✓ Generated {len(all_texts)} embedding texts from {len(processed_titles)} unique entities")
    
    # Save to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved to {output_path}")
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
    
    print(f"\n✓ Total texts generated: {len(texts_ko)}")


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
