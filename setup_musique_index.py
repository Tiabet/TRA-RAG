#!/usr/bin/env python3
"""
MuSiQue Pipeline Setup
======================
MuSiQue 데이터셋을 위한 인덱스 구축 파이프라인

Steps:
1. 메타데이터 JSON → SQLite DB 변환
2. DB → embedding_texts.json 생성
3. BM25 인덱스 구축
4. Dense 인덱스 구축

Usage:
    python setup_musique_index.py
"""

import json
import sqlite3
from pathlib import Path


def convert_metadata_to_db(
    metadata_json_path: str,
    db_path: str
):
    """
    메타데이터 JSON을 SQLite DB로 변환
    
    Args:
        metadata_json_path: 입력 메타데이터 JSON 경로
        db_path: 출력 SQLite DB 경로
    """
    print("="*60)
    print("Step 1: Converting Metadata JSON to SQLite DB")
    print("="*60)
    
    # Load metadata JSON
    print(f"\n📂 Loading: {metadata_json_path}")
    with open(metadata_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   ✓ Loaded {len(data)} items")
    
    # Create database
    print(f"\n💾 Creating DB: {db_path}")
    
    # Remove existing DB
    db_file = Path(db_path)
    if db_file.exists():
        db_file.unlink()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table (same schema as HotpotQA)
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
    
    for item in data:
        for ctx_meta in item.get('context_metadata', []):
            title = ctx_meta.get('title', '')
            metadata = ctx_meta.get('metadata', {})
            
            if not title or not metadata:
                skipped += 1
                continue
            
            try:
                cursor.execute(
                    'INSERT OR REPLACE INTO metadata (title, metadata_json) VALUES (?, ?)',
                    (title, json.dumps(metadata, ensure_ascii=False))
                )
                inserted += 1
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
    
    print(f"\n📊 Results:")
    print(f"   Inserted: {inserted}")
    print(f"   Skipped: {skipped}")
    print(f"   Total in DB: {total}")
    print(f"   ✓ DB created: {db_path}")


def generate_embedding_texts(
    db_path: str,
    output_path: str
):
    """
    DB에서 embedding texts 생성
    """
    print("\n" + "="*60)
    print("Step 2: Generating Embedding Texts")
    print("="*60)
    
    from embedding_text_generator import EmbeddingTextGenerator
    
    generator = EmbeddingTextGenerator(language="en")
    
    # Load from DB
    print(f"\n📂 Loading from: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT title, metadata_json FROM metadata')
    rows = cursor.fetchall()
    print(f"   ✓ Loaded {len(rows)} entries")
    
    # Generate embedding texts
    print(f"\n🔄 Generating embedding texts...")
    all_texts = []
    
    for row in rows:
        title = row['title']
        metadata = json.loads(row['metadata_json'])
        
        # Wrap metadata for extraction
        wrapped = {'metadata': metadata, 'title': title}
        texts = generator.extract_embedding_texts(title, wrapped)
        
        for t in texts:
            all_texts.append({
                'title': title,
                'key_path': t['key_path'],
                'value': str(t['value']),
                'text': t['text']
            })
    
    conn.close()
    
    # Save
    print(f"\n💾 Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    
    print(f"   ✓ Generated {len(all_texts)} embedding texts")
    print(f"   ✓ Unique titles: {len(set(t['title'] for t in all_texts))}")


def build_bm25_index(
    embedding_texts_path: str,
    index_save_path: str
):
    """
    BM25 인덱스 구축
    """
    print("\n" + "="*60)
    print("Step 3: Building BM25 Index")
    print("="*60)
    
    from bm25_indexer import BM25Indexer
    
    indexer = BM25Indexer(use_stemming=True)
    indexer.build_index(
        embedding_texts_path=embedding_texts_path,
        index_save_path=index_save_path
    )


def build_dense_index(
    embedding_texts_path: str,
    index_save_path: str
):
    """
    Dense (벡터) 인덱스 구축
    """
    print("\n" + "="*60)
    print("Step 4: Building Dense Index")
    print("="*60)
    
    from path_embedding_generator import PathEmbeddingGenerator
    import asyncio
    
    generator = PathEmbeddingGenerator()
    asyncio.run(generator.generate_embeddings(
        input_path=embedding_texts_path,
        output_path=index_save_path
    ))


def main():
    """MuSiQue 인덱스 전체 구축"""
    
    # Paths
    MUSIQUE_DIR = Path("MuSiQue")
    
    METADATA_JSON = MUSIQUE_DIR / "musique_sample_200_metadata.json"
    DB_PATH = MUSIQUE_DIR / "metadata_v3.db"
    EMBEDDING_TEXTS = MUSIQUE_DIR / "embedding_texts.json"
    BM25_INDEX = MUSIQUE_DIR / "bm25_index"
    DENSE_INDEX = MUSIQUE_DIR / "path_embeddings.npz"
    
    print("\n" + "="*80)
    print("🚀 MuSiQue Index Build Pipeline")
    print("="*80)
    print(f"\n📁 Working directory: {MUSIQUE_DIR}")
    print(f"   Metadata JSON: {METADATA_JSON}")
    print(f"   Output DB: {DB_PATH}")
    print(f"   Embedding texts: {EMBEDDING_TEXTS}")
    print(f"   BM25 index: {BM25_INDEX}")
    print(f"   Dense index: {DENSE_INDEX}")
    
    # Step 1: Convert metadata to DB
    convert_metadata_to_db(
        metadata_json_path=str(METADATA_JSON),
        db_path=str(DB_PATH)
    )
    
    # Step 2: Generate embedding texts
    generate_embedding_texts(
        db_path=str(DB_PATH),
        output_path=str(EMBEDDING_TEXTS)
    )
    
    # Step 3: Build BM25 index
    build_bm25_index(
        embedding_texts_path=str(EMBEDDING_TEXTS),
        index_save_path=str(BM25_INDEX)
    )
    
    # Step 4: Build Dense index
    build_dense_index(
        embedding_texts_path=str(EMBEDDING_TEXTS),
        index_save_path=str(DENSE_INDEX)
    )
    
    print("\n" + "="*80)
    print("✅ MuSiQue Index Build Complete!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {DB_PATH}")
    print(f"  - {EMBEDDING_TEXTS}")
    print(f"  - {BM25_INDEX}/")
    print(f"  - {DENSE_INDEX}/")
    print("\n")


if __name__ == "__main__":
    main()
