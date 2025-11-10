"""
Inspect Metadata Database
==========================
Examine the structure and content of the metadata database.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List
from collections import Counter


def inspect_database(db_path: str = 'HotpotQA/metadata_v2.db'):
    """
    Comprehensive inspection of the metadata database.
    """
    print("="*100)
    print(f"📊 Inspecting Metadata Database: {db_path}")
    print("="*100)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 1. Database Schema
    print("\n" + "="*100)
    print("1. DATABASE SCHEMA")
    print("="*100)
    
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    
    for table in tables:
        if table[0]:
            print(f"\n{table[0]}\n")
    
    # 2. Table Statistics
    print("\n" + "="*100)
    print("2. TABLE STATISTICS")
    print("="*100)
    
    # Main metadata table
    cursor.execute("SELECT COUNT(*) FROM metadata")
    total_count = cursor.fetchone()[0]
    print(f"\n📦 Total Metadata Entries: {total_count:,}")
    
    # Type distribution
    cursor.execute("""
        SELECT type, COUNT(*) as count 
        FROM metadata 
        GROUP BY type 
        ORDER BY count DESC
    """)
    type_dist = cursor.fetchall()
    
    print("\n📊 Distribution by Type:")
    for row in type_dist:
        print(f"   {row['type']}: {row['count']:,} ({row['count']/total_count*100:.1f}%)")
    
    # Subtype distribution (top 20)
    cursor.execute("""
        SELECT type, subtype, COUNT(*) as count 
        FROM metadata 
        WHERE subtype IS NOT NULL
        GROUP BY type, subtype 
        ORDER BY count DESC
        LIMIT 20
    """)
    subtype_dist = cursor.fetchall()
    
    print("\n📊 Top 20 Subtype Distribution:")
    for row in subtype_dist:
        print(f"   {row['type']} / {row['subtype']}: {row['count']:,}")
    
    # 3. Sample Entries
    print("\n" + "="*100)
    print("3. SAMPLE ENTRIES (First 5)")
    print("="*100)
    
    cursor.execute("""
        SELECT id, title, type, subtype, metadata_json
        FROM metadata
        LIMIT 5
    """)
    samples = cursor.fetchall()
    
    for i, row in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"ID: {row['id']}")
        print(f"Title: {row['title']}")
        print(f"Type: {row['type']} / {row['subtype']}")
        
        metadata = json.loads(row['metadata_json'])
        print(f"\nMetadata Keys: {list(metadata.keys())}")
        
        # Show first few fields
        print("\nMetadata Preview:")
        for key, value in list(metadata.items())[:5]:
            if isinstance(value, str) and len(value) > 100:
                print(f"   {key}: {value[:100]}...")
            elif isinstance(value, (list, dict)):
                print(f"   {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"   {key}: {value}")
    
    # 4. Metadata Field Analysis
    print("\n" + "="*100)
    print("4. METADATA FIELD ANALYSIS")
    print("="*100)
    
    # Analyze all metadata fields
    cursor.execute("SELECT metadata_json FROM metadata")
    all_metadata = cursor.fetchall()
    
    field_counter = Counter()
    field_types = {}
    
    for row in all_metadata[:1000]:  # Sample first 1000 for performance
        metadata = json.loads(row['metadata_json'])
        for key, value in metadata.items():
            field_counter[key] += 1
            if key not in field_types:
                field_types[key] = type(value).__name__
    
    print(f"\n📋 Common Metadata Fields (from 1000 samples):")
    for field, count in field_counter.most_common(30):
        print(f"   {field}: {count} ({count/10:.1f}%) [{field_types[field]}]")
    
    # 5. Specific Type Examples
    print("\n" + "="*100)
    print("5. EXAMPLES BY TYPE")
    print("="*100)
    
    for type_name in ['Person', 'Location', 'Organization', 'WorkOfArt', 'Event']:
        cursor.execute("""
            SELECT title, subtype, metadata_json
            FROM metadata
            WHERE type = ?
            LIMIT 2
        """, (type_name,))
        
        examples = cursor.fetchall()
        
        if examples:
            print(f"\n--- {type_name} Examples ---")
            for ex in examples:
                print(f"\nTitle: {ex['title']}")
                print(f"Subtype: {ex['subtype']}")
                metadata = json.loads(ex['metadata_json'])
                print(f"Fields: {', '.join(list(metadata.keys())[:10])}")
                
                # Show description if available
                if 'description' in metadata:
                    desc = metadata['description']
                    if isinstance(desc, str):
                        print(f"Description: {desc[:150]}...")
    
    # 6. Search Index Info
    print("\n" + "="*100)
    print("6. FULL-TEXT SEARCH INDEX")
    print("="*100)
    
    cursor.execute("""
        SELECT COUNT(*) FROM metadata_fts
    """)
    fts_count = cursor.fetchone()[0]
    print(f"\nFTS Index Entries: {fts_count:,}")
    
    # Test FTS search
    test_query = "university"
    cursor.execute("""
        SELECT COUNT(*) FROM metadata_fts 
        WHERE metadata_fts MATCH ?
    """, (test_query,))
    match_count = cursor.fetchone()[0]
    print(f"\nTest Search '{test_query}': {match_count:,} matches")
    
    # 7. Database File Size
    print("\n" + "="*100)
    print("7. DATABASE FILE INFO")
    print("="*100)
    
    db_file = Path(db_path)
    if db_file.exists():
        size_mb = db_file.stat().st_size / (1024 * 1024)
        print(f"\nDatabase File Size: {size_mb:.2f} MB")
    
    # Vacuum info
    cursor.execute("PRAGMA page_count")
    page_count = cursor.fetchone()[0]
    cursor.execute("PRAGMA page_size")
    page_size = cursor.fetchone()[0]
    
    print(f"Pages: {page_count:,}")
    print(f"Page Size: {page_size:,} bytes")
    print(f"Total Size: {(page_count * page_size) / (1024*1024):.2f} MB")
    
    conn.close()
    
    print("\n" + "="*100)
    print("✅ Inspection Complete!")
    print("="*100)


def search_by_title(db_path: str, title_pattern: str):
    """
    Search metadata by title pattern.
    """
    print(f"\n🔍 Searching for title: {title_pattern}")
    print("="*100)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, title, type, subtype, metadata_json
        FROM metadata
        WHERE title LIKE ?
        LIMIT 10
    """, (f"%{title_pattern}%",))
    
    results = cursor.fetchall()
    
    print(f"\nFound {len(results)} results (showing up to 10):\n")
    
    for row in results:
        print(f"--- ID: {row['id']} ---")
        print(f"Title: {row['title']}")
        print(f"Type: {row['type']} / {row['subtype']}")
        
        metadata = json.loads(row['metadata_json'])
        print(f"Fields: {', '.join(metadata.keys())}")
        
        if 'description' in metadata:
            desc = metadata['description']
            if isinstance(desc, str):
                print(f"Description: {desc[:200]}...")
        print()
    
    conn.close()


def search_by_type(db_path: str, entity_type: str, subtype: str = None):
    """
    Search metadata by type and optionally subtype.
    """
    print(f"\n🔍 Searching for type: {entity_type}" + (f" / {subtype}" if subtype else ""))
    print("="*100)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if subtype:
        cursor.execute("""
            SELECT id, title, type, subtype, metadata_json
            FROM metadata
            WHERE type = ? AND subtype = ?
            LIMIT 10
        """, (entity_type, subtype))
    else:
        cursor.execute("""
            SELECT id, title, type, subtype, metadata_json
            FROM metadata
            WHERE type = ?
            LIMIT 10
        """, (entity_type,))
    
    results = cursor.fetchall()
    
    print(f"\nFound {len(results)} results (showing up to 10):\n")
    
    for row in results:
        print(f"--- {row['title']} ---")
        print(f"Type: {row['type']} / {row['subtype']}")
        
        metadata = json.loads(row['metadata_json'])
        
        # Show key fields
        for key in ['description', 'main_entity', 'attributes', 'events']:
            if key in metadata:
                value = metadata[key]
                if isinstance(value, str):
                    print(f"{key}: {value[:150]}...")
                elif isinstance(value, dict):
                    print(f"{key}: {list(value.keys())[:5]}")
                elif isinstance(value, list):
                    print(f"{key}: {len(value)} items")
        print()
    
    conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect metadata database")
    parser.add_argument('--db', default='HotpotQA/metadata_v2.db', help='Database path')
    parser.add_argument('--search-title', help='Search by title pattern')
    parser.add_argument('--search-type', help='Search by entity type')
    parser.add_argument('--search-subtype', help='Search by subtype (requires --search-type)')
    
    args = parser.parse_args()
    
    if args.search_title:
        search_by_title(args.db, args.search_title)
    elif args.search_type:
        search_by_type(args.db, args.search_type, args.search_subtype)
    else:
        # Default: full inspection
        inspect_database(args.db)
