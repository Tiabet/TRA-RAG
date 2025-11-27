"""
Initialize Metadata Database
=============================
Create and populate the metadata database from JSON file.
"""

import json
from metadata_db import MetadataDB
import os

def initialize_database(
    metadata_file='HotpotQA/hotpotqa_sample_200_pure_metadata_v2.json',
    db_path='HotpotQA/metadata_v2.db',
    recreate=False
):
    """
    Initialize database from metadata JSON file.
    
    Args:
        metadata_file: Path to metadata JSON file
        db_path: Path to SQLite database
        recreate: If True, delete existing database first
    """
    print("="*100)
    print("📊 Initializing Metadata Database")
    print("="*100)
    
    # Recreate if requested
    if recreate and os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed existing database: {db_path}")
    
    # Load metadata
    print(f"\n📂 Loading metadata from: {metadata_file}")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    print(f"✅ Loaded {len(metadata_list):,} metadata entries")
    
    # Create database
    print(f"\n🗄️ Creating database: {db_path}")
    db = MetadataDB(db_path)
    db.create_tables()
    
    # Insert metadata
    print(f"\n💾 Inserting metadata into database...")
    db.insert_metadata_list(metadata_list)
    
    # Show statistics
    print(f"\n📊 Database Statistics:")
    stats = db.get_stats()
    print(f"   Total entries: {stats['total_entries']:,}")
    
    print(f"\n📊 Type distribution:")
    for entity_type, count in sorted(
        stats['type_distribution'].items(),
        key=lambda x: -x[1]
    )[:15]:
        percentage = count / stats['total_entries'] * 100
        print(f"   {entity_type:20s}: {count:5,} ({percentage:5.1f}%)")
    
    db.close()
    
    print(f"\n✅ Database initialization complete!")
    print("="*100)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize metadata database")
    parser.add_argument(
        '--metadata-file',
        default='HotpotQA/hotpotqa_sample_200_pure_metadata_v2.json',
        help='Path to metadata JSON file'
    )
    parser.add_argument(
        '--db-path',
        default='HotpotQA/metadata_v2.db',
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='Recreate database (delete existing)'
    )
    
    args = parser.parse_args()
    
    initialize_database(
        metadata_file=args.metadata_file,
        db_path=args.db_path,
        recreate=args.recreate
    )
