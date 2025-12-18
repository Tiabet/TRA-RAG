#!/usr/bin/env python3
"""\
HotpotQA Index Generation Pipeline
=================================
End-to-end pipeline for HotpotQA:
1) (Optional) create a sample file from HotpotQA/hotpotqa.json
2) Build LLM metadata JSON (context_metadata)
3) Convert metadata JSON -> SQLite DB (metadata_v4aligned.db)
4) Generate embedding_texts.json from DB
5) Generate dense path embeddings (path_embeddings.npz)
6) Build BM25 index from stopword-filtered embedding entry text

Defaults match the doc_id-aligned (v4aligned) artifacts.
"""

import argparse
import asyncio
import json
from pathlib import Path

from setup_hotpotqa_db import convert_metadata_to_db
from embedding_text_generator import generate_embedding_texts_from_db
from path_embedding_generator import PathEmbeddingGenerator
from bm25_indexer import BM25Indexer

from build_metadata import initialize_llm_client, process_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HotpotQA metadata + indices")

    parser.add_argument("--data", type=str, default="HotpotQA/hotpotqa_sample_200.json", help="Input QA JSON")
    parser.add_argument("--metadata_json", type=str, default="HotpotQA/hotpotqa_sample_200_metadata_v4aligned.json", help="Output metadata JSON")
    parser.add_argument("--db", type=str, default="HotpotQA/metadata_v4aligned.db", help="Output SQLite DB")
    parser.add_argument("--embedding_texts", type=str, default="HotpotQA/embedding_texts_v4aligned.json", help="Output embedding texts")
    parser.add_argument("--embeddings", type=str, default="HotpotQA/path_embeddings_v4aligned.npz", help="Output dense embeddings")
    parser.add_argument("--bm25_index", type=str, default="HotpotQA/bm25_index_v4aligned", help="Output BM25 index dir")

    # Metadata build controls
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini", help="LLM model for metadata")
    parser.add_argument("--max_passages", type=int, default=None, help="Limit number of passages for metadata (smoke test)")
    parser.add_argument("--metadata_concurrency", type=int, default=50, help="Concurrent metadata calls")
    parser.add_argument("--metadata_batch_size", type=int, default=10, help="Intermediate save batch")

    # Embedding controls
    parser.add_argument("--embed_batch_size", type=int, default=200, help="Embedding batch size")
    parser.add_argument("--embed_concurrency", type=int, default=5, help="Embedding max concurrency")

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    metadata_path = Path(args.metadata_json)
    db_path = Path(args.db)
    embedding_texts_path = Path(args.embedding_texts)
    embeddings_path = Path(args.embeddings)
    bm25_index_path = Path(args.bm25_index)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_texts_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_index_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Build metadata JSON if missing
    if not metadata_path.exists():
        print("=" * 80)
        print("[Step 1] Building metadata JSON (LLM)")
        print("=" * 80)

        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        client = initialize_llm_client()
        results = await process_dataset(
            client=client,
            data=data,
            model=args.model,
            max_passages=args.max_passages,
            batch_size=args.metadata_batch_size,
            concurrency=args.metadata_concurrency,
            output_path=metadata_path,
        )
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved metadata JSON -> {metadata_path}")
    else:
        print(f"[Skip] Metadata JSON already exists: {metadata_path}")

    # 2) Metadata JSON -> DB
    print("\n" + "=" * 80)
    print("[Step 2] Converting metadata JSON -> SQLite DB")
    print("=" * 80)
    convert_metadata_to_db(
        metadata_json_path=str(metadata_path),
        db_path=str(db_path),
        dedup_by_title=False,
    )

    # 3) DB -> embedding_texts.json
    print("\n" + "=" * 80)
    print("[Step 3] Generating embedding_texts.json")
    print("=" * 80)
    generate_embedding_texts_from_db(
        db_path=str(db_path),
        output_path=str(embedding_texts_path),
        language="en",
    )

    # 4) Dense path embeddings
    print("\n" + "=" * 80)
    print("[Step 4] Generating dense path embeddings")
    print("=" * 80)
    generator = PathEmbeddingGenerator(batch_size=args.embed_batch_size, max_concurrency=args.embed_concurrency)
    await generator.generate_embeddings(
        input_path=str(embedding_texts_path),
        output_path=str(embeddings_path),
    )

    # 5) BM25 index
    print("\n" + "=" * 80)
    print("[Step 5] Building BM25 index")
    print("=" * 80)
    indexer = BM25Indexer(use_stemming=True)
    indexer.build_index(
        embedding_texts_path=str(embedding_texts_path),
        index_save_path=str(bm25_index_path),
        use_embedding_text_field=True,
        strip_stopwords_in_embedding_text=True,
    )

    print("\n[OK] HotpotQA indices generated successfully")


if __name__ == "__main__":
    asyncio.run(main())
