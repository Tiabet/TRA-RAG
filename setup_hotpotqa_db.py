import json
import sqlite3
from pathlib import Path

def convert_metadata_to_db(
    metadata_json_path: str,
    db_path: str
):
    """
    Convert Metadata JSON to SQLite DB
    
    Args:
        metadata_json_path: Path to input metadata JSON
        db_path: Path to output SQLite DB
    """
    print("="*60)
    print("Converting Metadata JSON to SQLite DB")
    print("="*60)
    
    # Load metadata JSON
    print(f"\nLoading: {metadata_json_path}")
    with open(metadata_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   ✓ Loaded {len(data)} QA items")
    
    # Create database
    print(f"\nCreating DB: {db_path}")
    
    # Remove existing DB if it exists to start fresh
    db_file = Path(db_path)
    if db_file.exists():
        print("   Removing existing DB file...")
        db_file.unlink()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT UNIQUE NOT NULL,
            metadata_json TEXT NOT NULL
        )
    ''')
    
    # Insert metadata
    inserted = 0
    skipped = 0
    processed_titles = set()
    
    print("\nInserting metadata entries...")
    
    for item in data:
        for ctx_meta in item.get('context_metadata', []):
            title = ctx_meta.get('title', '')
            # Sometimes metadata is wrapped in 'metadata' key, sometimes it's the object itself
            # In hotpotqa_sample_200_metadata.json, it seems to be:
            # { "title": "...", "metadata": { ... } }
            metadata = ctx_meta.get('metadata', ctx_meta)
            
            if not title:
                skipped += 1
                continue
            
            if title in processed_titles:
                continue
                
            try:
                cursor.execute(
                    'INSERT OR REPLACE INTO metadata (title, metadata_json) VALUES (?, ?)',
                    (title, json.dumps(metadata, ensure_ascii=False))
                )
                inserted += 1
                processed_titles.add(title)
            except Exception as e:
                print(f"   ⚠️ Error inserting {title}: {e}")
                skipped += 1
    
    conn.commit()
    
    # Create index
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_title ON metadata(title)')
    conn.commit()
    
    # Stats
    cursor.execute('SELECT COUNT(*) FROM metadata')
    total = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\nResults:")
    print(f"   Total Unique Entries: {total}")
    print(f"   Inserted: {inserted}")
    print(f"   Skipped (duplicates/empty): {skipped}")
    print(f"   DB Saved to: {db_path}")

if __name__ == "__main__":
    convert_metadata_to_db(
        metadata_json_path='HotpotQA/hotpotqa_sample_200_metadata.json',
        db_path='HotpotQA/metadata_v3.db'
    )
