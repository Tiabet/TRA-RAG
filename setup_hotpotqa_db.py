import json
import sqlite3
from pathlib import Path

def convert_metadata_to_db(
    metadata_json_path: str,
    db_path: str,
    dedup_by_title: bool = True
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
    print(f"   [OK] Loaded {len(data)} QA items")
    
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
    # We intentionally do NOT use title as a unique key.
    # Many datasets (e.g., MuSiQue) contain repeated titles with different metadata.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            doc_id TEXT PRIMARY KEY,
            source_title TEXT NOT NULL,
            entity_title TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
    ''')
    
    # Insert metadata
    inserted = 0
    skipped = 0
    processed_titles = set()
    
    print("\nInserting metadata entries...")
    
    for qi, item in enumerate(data):
        qid = item.get('_id') or item.get('id') or str(qi)
        for ci, ctx_meta in enumerate(item.get('context_metadata', [])):
            # Sometimes metadata is wrapped in 'metadata' key, sometimes it's the object itself
            # In some datasets (e.g., LVEVAL), ctx_meta['title'] can be empty while
            # the actual title is stored under ctx_meta['metadata']['title'].
            metadata = ctx_meta.get('metadata', ctx_meta)

            raw_source_title = ctx_meta.get('title', '')
            raw_entity_title = (metadata or {}).get('title', '')

            # Prefer metadata.title as the canonical title when available.
            source_title = raw_entity_title or raw_source_title
            entity_title = raw_entity_title or raw_source_title
            
            if dedup_by_title and entity_title in processed_titles:
                continue

            # Prefer stable doc_id produced from the ORIGINAL dataset context index.
            # build_metadata.py now writes doc_id/ctx_idx explicitly to keep ordering stable
            # even under async execution.
            doc_id = ctx_meta.get('doc_id')
            if not doc_id:
                ctx_idx = ctx_meta.get('ctx_idx')
                if ctx_idx is not None:
                    doc_id = f"{qid}::ctx{int(ctx_idx)}"
                else:
                    doc_id = f"{qid}::ctx{ci}"

            if not source_title:
                # Keep DB schema constraints (NOT NULL) while still allowing insertion.
                source_title = f"doc_{doc_id}"
                if not entity_title:
                    entity_title = source_title
                
            try:
                cursor.execute(
                    'INSERT OR REPLACE INTO metadata (doc_id, source_title, entity_title, metadata_json) VALUES (?, ?, ?, ?)',
                    (doc_id, source_title, entity_title, json.dumps(metadata, ensure_ascii=False))
                )
                inserted += 1
                if dedup_by_title:
                    processed_titles.add(entity_title)
            except Exception as e:
                print(f"   [WARN] Error inserting {source_title}: {e}")
                skipped += 1
    
    conn.commit()
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_title ON metadata(entity_title)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_title ON metadata(source_title)')
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
        metadata_json_path='HotpotQA/hotpotqa_sample_200_metadata_v4aligned.json',
        db_path='HotpotQA/metadata_v4aligned.db'
    )
