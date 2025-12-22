#!/usr/bin/env python3
"""HotpotQA v5 (corpus_idx) Index Generation Pipeline

This builds fresh v5 artifacts from the corpus_idx-aware metadata JSON.

Artifacts:
- HotpotQA/metadata_v5.db
- HotpotQA/embedding_texts_v5.json
- HotpotQA/path_embeddings_v5.npz
- HotpotQA/bm25_index_v5/

Notes:
- Expects metadata JSON where each context_metadata entry has doc_id (corpus_idx).
- Uses EmbeddingTextGenerator which now splits relations.target lists into per-target facts.
"""

import argparse
import asyncio
from pathlib import Path

from setup_hotpotqa_db import convert_metadata_to_db
from embedding_text_generator import generate_embedding_texts_from_db
from path_embedding_generator import PathEmbeddingGenerator
from bm25_indexer import BM25Indexer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HotpotQA v5 indices (corpus_idx)")
    p.add_argument(
        "--metadata_json",
        type=str,
        default="HotpotQA/hotpotqa_sample_200_corpus_idx_metadata.json",
        help="Input metadata JSON (context_metadata)",
    )
    p.add_argument("--db", type=str, default="HotpotQA/metadata_v5.db", help="Output SQLite DB")
    p.add_argument(
        "--embedding_texts",
        type=str,
        default="HotpotQA/embedding_texts_v5.json",
        help="Output embedding texts JSON",
    )
    p.add_argument(
        "--embeddings",
        type=str,
        default="HotpotQA/path_embeddings_v5.npz",
        help="Output dense path embeddings (.npz)",
    )
    p.add_argument(
        "--bm25_index",
        type=str,
        default="HotpotQA/bm25_index_v5",
        help="Output BM25 index directory",
    )
    p.add_argument("--embed_batch_size", type=int, default=200, help="Embedding batch size")
    p.add_argument("--embed_concurrency", type=int, default=5, help="Embedding max concurrency")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    metadata_path = Path(args.metadata_json)
    db_path = Path(args.db)
    embedding_texts_path = Path(args.embedding_texts)
    embeddings_path = Path(args.embeddings)
    bm25_index_path = Path(args.bm25_index)

    print("Starting HotpotQA v5 Index Generation Pipeline...")

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata JSON not found: {metadata_path}. Run build_metadata.py first (or pass --metadata_json)."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_texts_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_index_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Metadata JSON -> DB
    print("\n[Step 1] Converting metadata JSON -> SQLite DB...")
    convert_metadata_to_db(
        metadata_json_path=str(metadata_path),
        db_path=str(db_path),
        dedup_by_title=False,
    )

    # 2) DB -> embedding_texts.json
    print("\n[Step 2] Generating embedding_texts.json...")
    generate_embedding_texts_from_db(
        db_path=str(db_path),
        output_path=str(embedding_texts_path),
        language="en",
    )

    # 3) Dense embeddings
    print("\n[Step 3] Generating dense path embeddings...")
    generator = PathEmbeddingGenerator(batch_size=args.embed_batch_size, max_concurrency=args.embed_concurrency)
    await generator.generate_embeddings(
        input_path=str(embedding_texts_path),
        output_path=str(embeddings_path),
    )

    # 4) BM25 index
    print("\n[Step 4] Building BM25 index...")
    indexer = BM25Indexer(use_stemming=True)
    indexer.build_index(
        embedding_texts_path=str(embedding_texts_path),
        index_save_path=str(bm25_index_path),
        use_embedding_text_field=True,
        strip_stopwords_in_embedding_text=True,
    )

    print("\n[OK] HotpotQA v5 indices generated successfully")


if __name__ == "__main__":
    asyncio.run(main())
