import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

from setup_hotpotqa_db import convert_metadata_to_db
from embedding_text_generator import generate_embedding_texts_from_db
from path_embedding_generator import PathEmbeddingGenerator
from bm25_indexer import BM25Indexer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MuSiQue indices")

    parser.add_argument(
        "--metadata_json",
        type=str,
        default="MuSiQue/musique_sample_200_metadata_v4aligned.json",
        help="Input metadata JSON (context_metadata)",
    )
    parser.add_argument("--db", type=str, default="MuSiQue/metadata_v4aligned.db", help="Output SQLite DB")
    parser.add_argument(
        "--embedding_texts",
        type=str,
        default="MuSiQue/embedding_texts_v4aligned.json",
        help="Output embedding texts JSON",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="MuSiQue/path_embeddings_v4aligned.npz",
        help="Output dense path embeddings (.npz)",
    )
    parser.add_argument(
        "--bm25_index",
        type=str,
        default="MuSiQue/bm25_index_v4aligned",
        help="Output BM25 index directory",
    )

    parser.add_argument("--embed_batch_size", type=int, default=200, help="Embedding batch size")
    parser.add_argument("--embed_concurrency", type=int, default=5, help="Embedding max concurrency")

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    metadata_path = Path(args.metadata_json)
    db_path = Path(args.db)
    embedding_texts_path = Path(args.embedding_texts)
    embeddings_path = Path(args.embeddings)
    bm25_index_path = Path(args.bm25_index)

    print("Starting MuSiQue Index Generation Pipeline...")

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata JSON not found: {metadata_path}. Run build_metadata.py first (or pass --metadata_json)."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_texts_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_index_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Generate DB
    print("\n[Step 1] Generating Metadata DB...")
    convert_metadata_to_db(
        metadata_json_path=str(metadata_path),
        db_path=str(db_path),
        dedup_by_title=False,
    )

    # 2) Generate Embedding Texts
    print("\n[Step 2] Generating Embedding Texts...")
    generate_embedding_texts_from_db(
        db_path=str(db_path),
        output_path=str(embedding_texts_path),
        language="en",
    )

    # 3) Generate Dense Embeddings
    print("\n[Step 3] Generating Dense Embeddings...")
    generator = PathEmbeddingGenerator(batch_size=args.embed_batch_size, max_concurrency=args.embed_concurrency)
    await generator.generate_embeddings(
        input_path=str(embedding_texts_path),
        output_path=str(embeddings_path),
    )

    # 4) Generate BM25 Index
    print("\n[Step 4] Generating BM25 Index...")
    indexer = BM25Indexer(use_stemming=True)
    indexer.build_index(
        embedding_texts_path=str(embedding_texts_path),
        index_save_path=str(bm25_index_path),
        use_embedding_text_field=True,
        strip_stopwords_in_embedding_text=True,
    )

    print("\nAll MuSiQue indices generated successfully!")


if __name__ == "__main__":
    asyncio.run(main())
