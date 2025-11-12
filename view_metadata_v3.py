#!/usr/bin/env python3
"""
View contents of metadata_v3.db (path-based storage)
"""

import sqlite3
import json

def view_database():
    """View database contents"""
    
    db_path = 'HotpotQA/metadata_v3.db'
    
    print("="*80)
    print("Metadata Database V3 Viewer (Path-based Storage)")
    print("="*80)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Database statistics
    print("\n[1] Database Statistics")
    print("-" * 80)
    
    cursor.execute("SELECT COUNT(*) as count FROM metadata")
    total = cursor.fetchone()['count']
    print(f"Total entries: {total}")
    
    # Type distribution
    cursor.execute("""
        SELECT type, subtype, COUNT(*) as count
        FROM metadata
        GROUP BY type, subtype
        ORDER BY count DESC
        LIMIT 15
    """)
    
    print("\nTop 15 Type-Subtype combinations:")
    for row in cursor.fetchall():
        print(f"  {row['type']:20s} - {row['subtype']:25s}: {row['count']:4d}")
    
    # Sample entries
    print("\n" + "="*80)
    print("[2] Sample Entries (First 3)")
    print("="*80)
    
    cursor.execute("""
        SELECT title, type, subtype, searchable_paths, metadata_json
        FROM metadata
        LIMIT 3
    """)
    
    for idx, row in enumerate(cursor.fetchall(), 1):
        print(f"\n{'-'*80}")
        print(f"Sample {idx}: {row['title']}")
        print(f"Type: {row['type']} - {row['subtype']}")
        print(f"\n[Original Metadata]")
        metadata = json.loads(row['metadata_json'])
        print(json.dumps(metadata, indent=2, ensure_ascii=False)[:500] + "...")
        
        print(f"\n[Extracted Paths - Total: {len(row['searchable_paths'].split(', '))}]")
        paths = row['searchable_paths'].split(', ')
        for i, path in enumerate(paths[:10], 1):
            print(f"  {i:2d}. {path}")
        if len(paths) > 10:
            print(f"  ... and {len(paths) - 10} more paths")
    
    # Search test
    print("\n" + "="*80)
    print("[3] Search Test - 'Leonardo DiCaprio'")
    print("="*80)
    
    cursor.execute("""
        SELECT m.title, m.type, m.subtype
        FROM metadata m
        JOIN metadata_fts f ON m.rowid = f.rowid
        WHERE f.searchable_paths MATCH 'leonardo AND dicaprio'
        LIMIT 5
    """)
    
    results = cursor.fetchall()
    print(f"\nFound {len(results)} results:")
    for row in results:
        print(f"  - {row['title']} ({row['type']} - {row['subtype']})")
    
    # Another search test
    print("\n" + "="*80)
    print("[4] Search Test - 'Argentina'")
    print("="*80)
    
    cursor.execute("""
        SELECT m.title, m.type, m.subtype
        FROM metadata m
        JOIN metadata_fts f ON m.rowid = f.rowid
        WHERE f.searchable_paths MATCH 'argentina'
        LIMIT 5
    """)
    
    results = cursor.fetchall()
    print(f"\nFound {len(results)} results:")
    for row in results:
        print(f"  - {row['title']} ({row['type']} - {row['subtype']})")
    
    # Path pattern analysis
    print("\n" + "="*80)
    print("[5] Path Pattern Analysis")
    print("="*80)
    
    cursor.execute("""
        SELECT searchable_paths
        FROM metadata
        WHERE title LIKE '%Wolf%'
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    if result:
        paths = result['searchable_paths'].split(', ')
        print(f"\nExample: Entry with 'Wolf' in title has {len(paths)} paths")
        print("\nAll paths:")
        for i, path in enumerate(paths, 1):
            print(f"  {i:2d}. {path}")
    
    conn.close()
    
    print("\n" + "="*80)
    print("✓ Database viewing complete!")
    print("="*80)

if __name__ == "__main__":
    view_database()
