"""
Rebuild metadata database with v2 metadata
"""
import os
import json
from metadata_db import MetadataDB

# Remove old database
db_path = 'metadata_v2.db'
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"✓ Removed old database: {db_path}")

# Initialize new database
print(f"\n{'='*60}")
print("Creating new database with v2 metadata")
print(f"{'='*60}")

db = MetadataDB(db_path)
db.create_tables()

# Load v2 metadata
print("\nLoading v2 metadata...")
with open('HotpotQA/hotpotqa_sample_200_pure_metadata_v2.json', 'r', encoding='utf-8') as f:
    metadata_list = json.load(f)

print(f"Loaded {len(metadata_list)} metadata entries")

# Insert into database
db.insert_metadata_list(metadata_list)

# Show statistics
print("\n" + "="*60)
print("Database Statistics")
print("="*60)
stats = db.get_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"\nType distribution:")
for entity_type, count in sorted(stats['type_distribution'].items(), key=lambda x: -x[1])[:10]:
    print(f"  {entity_type}: {count}")

db.close()

print("\n" + "="*60)
print(f"✓ Database created: {db_path}")
print("="*60)
