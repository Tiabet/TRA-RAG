"""
View Detailed Metadata
=======================
Display full metadata structure for specific entities.
"""

import sqlite3
import json
from pathlib import Path

def view_metadata_details(db_path: str, title_pattern: str):
    """
    View detailed metadata for entities matching title pattern.
    """
    print("="*100)
    print(f"🔍 Detailed Metadata View: {title_pattern}")
    print("="*100)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, title, type, subtype, metadata_json
        FROM metadata
        WHERE title LIKE ?
        LIMIT 3
    """, (f"%{title_pattern}%",))
    
    results = cursor.fetchall()
    
    if not results:
        print(f"\n❌ No results found for '{title_pattern}'")
        conn.close()
        return
    
    print(f"\nFound {len(results)} results (showing up to 3):\n")
    
    for i, row in enumerate(results, 1):
        print("="*100)
        print(f"ENTITY {i}: {row['title']}")
        print("="*100)
        print(f"ID: {row['id']}")
        print(f"Type: {row['type']} / {row['subtype']}")
        
        metadata = json.loads(row['metadata_json'])
        
        print(f"\n📋 Full Metadata Structure:")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        
        # Analyze structure
        print(f"\n📊 Metadata Analysis:")
        print(f"   Total Keys: {len(metadata)}")
        
        if 'attributes' in metadata:
            attrs = metadata['attributes']
            print(f"   Attributes: {len(attrs)} items")
            if attrs:
                print(f"      Keys: {list(attrs.keys())}")
        
        if 'relations' in metadata:
            rels = metadata['relations']
            print(f"   Relations: {len(rels)} items")
            if rels:
                for j, rel in enumerate(rels[:5], 1):
                    if isinstance(rel, dict):
                        print(f"      Relation {j}: {rel.get('relation_type', 'unknown')} -> {rel.get('entity', 'unknown')}")
        
        print()
    
    conn.close()


def view_type_examples(db_path: str, entity_type: str, subtype: str = None):
    """
    View examples of a specific type/subtype.
    """
    print("="*100)
    print(f"🔍 Type Examples: {entity_type}" + (f" / {subtype}" if subtype else ""))
    print("="*100)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if subtype:
        cursor.execute("""
            SELECT id, title, type, subtype, metadata_json
            FROM metadata
            WHERE type = ? AND subtype = ?
            LIMIT 3
        """, (entity_type, subtype))
    else:
        cursor.execute("""
            SELECT id, title, type, subtype, metadata_json
            FROM metadata
            WHERE type = ?
            LIMIT 3
        """, (entity_type,))
    
    results = cursor.fetchall()
    
    if not results:
        print(f"\n❌ No results found")
        conn.close()
        return
    
    print(f"\nFound {len(results)} examples:\n")
    
    for i, row in enumerate(results, 1):
        print("="*100)
        print(f"EXAMPLE {i}: {row['title']}")
        print("="*100)
        print(f"Type: {row['type']} / {row['subtype']}")
        
        metadata = json.loads(row['metadata_json'])
        print(f"\n📋 Full Metadata:")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print()
    
    conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View detailed metadata")
    parser.add_argument('--db', default='HotpotQA/metadata_v2.db', help='Database path')
    parser.add_argument('--title', help='Search by title pattern')
    parser.add_argument('--type', help='View examples by type')
    parser.add_argument('--subtype', help='View examples by subtype (requires --type)')
    
    args = parser.parse_args()
    
    if args.title:
        view_metadata_details(args.db, args.title)
    elif args.type:
        view_type_examples(args.db, args.type, args.subtype)
    else:
        print("Usage:")
        print("  --title <pattern>        : View metadata for entities matching title")
        print("  --type <type>            : View examples of specific type")
        print("  --type <type> --subtype <subtype> : View examples of specific type/subtype")
        print("\nExamples:")
        print("  python view_metadata.py --title 'university'")
        print("  python view_metadata.py --type Person --subtype Athlete")
        print("  python view_metadata.py --type WorkOfArt --subtype Film")
