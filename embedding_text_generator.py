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
            self._extract_recursive(
                title=title,
                obj=inner_metadata['attributes'],
                key_path=[],
                results=results
            )
        
        # Extract from relations
        if 'relations' in inner_metadata:
            for relation in inner_metadata.get('relations', []):
                self._extract_relation(title, relation, results)
        
        # Extract top-level fields (excluding type, subtype, title, attributes, relations)
        skip_keys = {'type', 'subtype', 'title', 'attributes', 'relations', 'metadata'}
        for key, value in inner_metadata.items():
            if key in skip_keys:
                continue
            self._extract_recursive(
                title=title,
                obj=value,
                key_path=[key],
                results=results
            )
        
        return results
    
    def _extract_recursive(
        self,
        title: str,
        obj: Any,
        key_path: List[str],
        results: List[Dict]
    ):
        """
        Recursively extract key-value pairs.
        
        Args:
            title: Entity title
            obj: Current object to process
            key_path: Current path of keys
            results: Accumulator for results
        """
        if obj is None:
            return
        
        # Skip type/subtype anywhere in the structure
        if isinstance(obj, dict):
            for key, value in obj.items():
                # Skip meaningless keys
                if key in ('type', 'subtype'):
                    continue
                
                new_path = key_path + [key]
                
                if isinstance(value, dict):
                    # Check if it's an entity object (has name/title as primary identifier)
                    if 'name' in value or 'title' in value:
                        # Extract the entity reference as a single value
                        entity_name = value.get('name') or value.get('title')
                        self._add_result(title, new_path, entity_name, results)
                        
                        # Also extract nested attributes of this entity
                        for sub_key, sub_value in value.items():
                            if sub_key in ('type', 'subtype', 'name', 'title'):
                                continue
                            sub_path = new_path + [sub_key]
                            self._extract_recursive(title, sub_value, sub_path, results)
                    else:
                        # Regular nested dict
                        self._extract_recursive(title, value, new_path, results)
                        
                elif isinstance(value, list):
                    # Process list items
                    for item in value:
                        if isinstance(item, dict):
                            # Entity in list
                            if 'name' in item or 'title' in item:
                                entity_name = item.get('name') or item.get('title')
                                self._add_result(title, new_path, entity_name, results)
                                
                                # Extract nested attributes
                                for sub_key, sub_value in item.items():
                                    if sub_key in ('type', 'subtype', 'name', 'title'):
                                        continue
                                    sub_path = new_path + [sub_key]
                                    self._extract_recursive(title, sub_value, sub_path, results)
                            else:
                                self._extract_recursive(title, item, new_path, results)
                        else:
                            # Simple value in list
                            self._add_result(title, new_path, item, results)
                else:
                    # Leaf value
                    self._add_result(title, new_path, value, results)
        
        elif isinstance(obj, list):
            for item in obj:
                self._extract_recursive(title, item, key_path, results)
        
        else:
            # Leaf value (shouldn't reach here normally, but handle it)
            if key_path:
                self._add_result(title, key_path, obj, results)
    
    def _extract_relation(self, title: str, relation: Dict, results: List[Dict]):
        """
        Extract from a relation object.
        
        Relation structure:
        {
            "relation": "directed_by",
            "target": {"name": "Martin Scorsese", "type": "Person", ...},
            "year": 2013,
            ...
        }
        Or target can be a list.
        """
        relation_type = relation.get('relation', 'related_to')
        target = relation.get('target', {})
        
        # Handle target as list or dict
        if isinstance(target, list):
            for t in target:
                if isinstance(t, dict):
                    self._extract_single_target(title, relation_type, t, results)
        elif isinstance(target, dict):
            self._extract_single_target(title, relation_type, target, results)
        
        # Extract other relation attributes (year, etc.)
        for key, value in relation.items():
            if key in ('relation', 'target'):
                continue
            key_path = [relation_type, key]
            self._extract_recursive(title, value, key_path, results)
    
    def _extract_single_target(self, title: str, relation_type: str, target: Dict, results: List[Dict]):
        """Extract from a single target entity."""
        target_name = target.get('name') or target.get('title')
        if target_name:
            self._add_result(title, [relation_type], target_name, results)
        
        # Extract target's nested attributes
        for key, value in target.items():
            if key in ('type', 'subtype', 'name', 'title'):
                continue
            key_path = [relation_type, key]
            self._extract_recursive(title, value, key_path, results)
    
    def _add_result(
        self,
        title: str,
        key_path: List[str],
        value: Any,
        results: List[Dict]
    ):
        """Add a result to the accumulator."""
        if value is None:
            return
        
        # Convert value to string
        value_str = str(value)
        
        # Skip empty values
        if not value_str.strip():
            return
        
        # Build key path string
        key_path_str = ".".join(key_path)
        
        # Generate natural language text
        if self.language == "ko":
            text = self._format_korean(title, key_path, value_str)
        else:
            text = self._format_english(title, key_path, value_str)
        
        results.append({
            'text': text,
            'key_path': key_path_str,
            'value': value_str,
            'title': title
        })
    
    def _format_korean(self, title: str, key_path: List[str], value: str) -> str:
        """Format as Korean natural language."""
        if len(key_path) == 1:
            return f"{title}의 {key_path[0]}은/는 {value}이다"
        else:
            path_str = "의 ".join(key_path)
            return f"{title}의 {path_str}은/는 {value}이다"
    
    def _format_english(self, title: str, key_path: List[str], value: str) -> str:
        """Format as English natural language."""
        if len(key_path) == 1:
            return f"The {key_path[0]} of {title} is {value}"
        else:
            # "The name of the director of Title is Value"
            path_str = " of the ".join(reversed(key_path))
            return f"The {path_str} of {title} is {value}"


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
    print("Generating Embedding Texts from Metadata")
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
    
    # Statistics
    print(f"\n[Statistics]")
    print(f"  Total entries: {len(all_texts)}")
    print(f"  Unique titles: {len(set(t['title'] for t in all_texts))}")
    
    # Sample output
    print(f"\n[Sample Outputs - First 10]")
    for i, text_entry in enumerate(all_texts[:10], 1):
        print(f"  {i}. {text_entry['text'][:100]}...")
    
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
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_single_metadata()
    else:
        # Generate from database (English only)
        generate_embedding_texts_from_db(
            db_path='HotpotQA/metadata_v3.db',
            output_path='HotpotQA/embedding_texts.json',
            language="en"
        )
