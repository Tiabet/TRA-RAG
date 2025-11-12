#!/usr/bin/env python3
"""
Initialize metadata_v3.db with path-based storage from hotpotqa metadata.
"""

import json
import os
from metadata_db_v2 import MetadataDBV2

def main():
    """Initialize the metadata database V2"""
    
    # Paths
    metadata_file = 'HotpotQA/hotpotqa_sample_200_pure_metadata_v2.json'
    db_path = 'HotpotQA/metadata_v3.db'
    
    print("="*80)
    print("Initializing Metadata Database V3 (Path-based Storage)")
    print("="*80)
    
    # Load metadata
    print(f"\n1. Loading metadata from: {metadata_file}")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)
    
    print(f"   ✓ Loaded {len(all_metadata)} metadata entries")
    
    # Remove existing database
    if os.path.exists(db_path):
        print(f"\n2. Removing existing database: {db_path}")
        os.remove(db_path)
        print(f"   ✓ Removed")
    
    # Create database
    print(f"\n3. Creating new database: {db_path}")
    with MetadataDBV2(db_path) as db:
        db.create_tables()
        print(f"   ✓ Tables created")
        
        # Prepare entries
        print(f"\n4. Preparing metadata entries...")
        metadata_entries = []
        for entry in all_metadata:
            title = entry.get('title', 'Unknown')
            metadata = entry.get('metadata', entry)
            
            metadata_entries.append({
                'title': title,
                'metadata': metadata
            })
        
        print(f"   ✓ Prepared {len(metadata_entries)} entries")
        
        # Bulk insert
        print(f"\n5. Inserting metadata into database...")
        db.insert_metadata_list(metadata_entries)
        
        # Get count
        db.cursor.execute("SELECT COUNT(*) as count FROM metadata")
        inserted = db.cursor.fetchone()['count']
        
        print(f"   ✓ Inserted: {inserted}")
        
        # Get statistics
        print(f"\n6. Database statistics:")
        print(f"   Total entries: {inserted}")
        
        # Type distribution
        db.cursor.execute("""
            SELECT type, COUNT(*) as count
            FROM metadata
            WHERE type IS NOT NULL
            GROUP BY type
            ORDER BY count DESC
            LIMIT 10
        """)
        type_counts = db.cursor.fetchall()
        
        if type_counts:
            print(f"   Top 10 entity types:")
            for row in type_counts:
                print(f"     - {row['type']}: {row['count']}")
    
    print("\n" + "="*80)
    print("✓ Database initialization complete!")
    print("="*80)
    print(f"\nDatabase location: {os.path.abspath(db_path)}")
    print(f"Total entries: {inserted}")

if __name__ == "__main__":
    main()
