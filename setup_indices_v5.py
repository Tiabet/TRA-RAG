#!/usr/bin/env python3
"""\
Unified v5 Index Generation Pipeline
===================================
Builds v5 artifacts end-to-end for supported datasets:
- LLM metadata JSON (context_metadata)
- SQLite DB (metadata_v5.db)
- embedding_texts_v5.json
- dense path embeddings (path_embeddings_v5.npz)
- BM25 index (bm25_index_v5/)

Examples:
  python setup_indices_v5.py --dataset hotpotqa
  python setup_indices_v5.py --dataset musique
  python setup_indices_v5.py --dataset 2wiki
  python setup_indices_v5.py --dataset lveval
  python setup_indices_v5.py --dataset all

Notes:
- Metadata build requires: ALICE_CHAT_URL, ALICE_OPENAI_KEY
- Embeddings require: ALICE_EMBED_URL, ALICE_OPENAI_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from bm25_indexer import BM25Indexer
from build_metadata import initialize_llm_client, process_dataset
from embedding_text_generator import generate_embedding_texts_from_db
from path_embedding_generator import PathEmbeddingGenerator
from llm_provider import detect_provider
from prepare_lveval_v5 import prepare_lveval_files
from setup_hotpotqa_db import convert_metadata_to_db


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    qa_path: str
    out_dir: str


DATASETS: dict[str, DatasetConfig] = {
    "nq_rear": DatasetConfig("nq_rear", "NQ_REAR/nq_rear_corpus_for_pipeline.json", "NQ_REAR"),
    "popqa": DatasetConfig("popqa", "POPQA/popqa_corpus_for_pipeline.json", "POPQA"),
    "hotpotqa": DatasetConfig(
        name="hotpotqa",
        qa_path="HotpotQA/hotpotqa.json",
        out_dir="HotpotQA",
    ),
    "musique": DatasetConfig(
        name="musique",
        qa_path="MuSiQue/musique.json",
        out_dir="MuSiQue",
    ),
    "2wiki": DatasetConfig(
        name="2wiki",
        qa_path="2WikiMultihopQA/2wikimultihopqa.json",
        out_dir="2WikiMultihopQA",
    ),
    "lveval": DatasetConfig(
        name="lveval",
        qa_path="LVEVAL/lveval_corpus_for_pipeline.json",
        out_dir="LVEVAL",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v5 metadata + indices for one or all datasets")

    p.add_argument("--dataset", type=str, default="all", choices=["all", *DATASETS.keys()])

    # Metadata build controls
    p.add_argument("--model", type=str, default=None, help="Default: model selected by llm_provider")
    p.add_argument("--fact_granularity", choices=['top_level', 'legacy_items'], default='top_level',
                   help="Published top-level Facts, or historical per-list-item indexing")
    p.add_argument("--metadata_concurrency", type=int, default=200)
    p.add_argument("--metadata_batch_size", type=int, default=10)
    p.add_argument("--max_passages", type=int, default=None)
    p.add_argument("--dry_run_metadata", action="store_true", help="Validate mapping, skip LLM calls")

    # Embedding controls
    p.add_argument("--embed_batch_size", type=int, default=200)
    p.add_argument("--embed_concurrency", type=int, default=10)

    # IO / overwrite
    p.add_argument("--rebuild", action="store_true", help="Overwrite existing v5 outputs")

    # Optional override (only meaningful for single-dataset runs)
    p.add_argument("--qa_path", type=str, default=None, help="Override QA input JSON path")

    return p.parse_args()


def _paths_for_dataset(cfg: DatasetConfig, root: Path) -> dict[str, Path]:
    out_dir = root / cfg.out_dir
    return {
        "qa": root / cfg.qa_path,
        "metadata_json": out_dir / "metadata_v5.json",
        "db": out_dir / "metadata_v5.db",
        "embedding_texts": out_dir / "embedding_texts_v5.json",
        "embeddings": out_dir / "path_embeddings_v5.npz",
        "bm25_index": out_dir / "bm25_index_v5",
    }


def validate_granularity(paths, manifest_path, requested):
    """Prevent silently mixing previously built Facts with a different policy."""
    existing = any(paths[k].exists() for k in ('embedding_texts', 'embeddings', 'bm25_index'))
    if not existing:
        return
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding='utf-8'))['fact_granularity']
    else:
        recorded = 'legacy_items'
    if recorded != requested:
        raise ValueError(
            f'Existing index granularity is {recorded}, requested {requested}. '
            'Use the matching --fact_granularity, or back up artifacts and rebuild with --rebuild.'
        )


async def build_for_dataset(args: argparse.Namespace, cfg: DatasetConfig, root: Path) -> None:
    paths = _paths_for_dataset(cfg, root)

    # LVEVAL has a separate global corpus. Avoid indexing the per-question giant `context` string.
    # Instead, generate a corpus wrapper (QA-like) file and build indices from it.
    if cfg.name == "lveval":
        lveval_paths = prepare_lveval_files(root, rebuild=bool(args.rebuild))
        paths["qa"] = lveval_paths["corpus_wrapper"]

    if args.qa_path and args.dataset != "all":
        paths["qa"] = (root / args.qa_path).resolve()

    qa_path = paths["qa"]
    metadata_json_path = paths["metadata_json"]
    db_path = paths["db"]
    embedding_texts_path = paths["embedding_texts"]
    embeddings_path = paths["embeddings"]
    bm25_index_path = paths["bm25_index"]
    manifest_path = embedding_texts_path.with_name('index_manifest.json')
    if not args.rebuild and not args.dry_run_metadata:
        validate_granularity(paths, manifest_path, args.fact_granularity)

    for p in [
        metadata_json_path.parent,
        db_path.parent,
        embedding_texts_path.parent,
        embeddings_path.parent,
        bm25_index_path.parent,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"[setup_indices_v5] dataset={cfg.name}")
    print(f"qa:             {qa_path}")
    print(f"metadata_json:   {metadata_json_path}")
    print(f"db:              {db_path}")
    print(f"embedding_texts:  {embedding_texts_path}")
    print(f"embeddings:       {embeddings_path}")
    print(f"bm25_index:       {bm25_index_path}")
    print("=" * 90)

    if not qa_path.exists():
        raise FileNotFoundError(f"QA JSON not found: {qa_path}")

    # 1) Metadata JSON (LLM)
    if metadata_json_path.exists() and not args.rebuild:
        print(f"[Skip] metadata JSON exists: {metadata_json_path}")
    else:
        print("\n[Step 1] Building metadata JSON (LLM)")
        with qa_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        client = None
        if not args.dry_run_metadata:
            client = initialize_llm_client()

        results = await process_dataset(
            client=client,  # unused when dry_run_metadata=True
            data=data,
            model=(args.model or ('gpt-4o-mini' if args.dry_run_metadata else detect_provider().chat_model)),
            max_passages=args.max_passages,
            batch_size=int(args.metadata_batch_size),
            concurrency=int(args.metadata_concurrency),
            output_path=metadata_json_path,
            dry_run=bool(args.dry_run_metadata),
        )

        # process_dataset may write incremental snapshots, but we always write final output.
        with metadata_json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved metadata JSON -> {metadata_json_path}")

    # Dry-run mode is only meant to validate doc_id/ctx_idx mapping and input parsing.
    # Stop early to avoid generating empty DB/embeddings.
    if args.dry_run_metadata:
        print(f"\n[OK] Dry-run metadata finished for {cfg.name}; skipping DB/embeddings/BM25 steps")
        return

    # 2) Metadata JSON -> DB
    if db_path.exists() and not args.rebuild:
        print(f"[Skip] DB exists: {db_path}")
    else:
        print("\n[Step 2] Converting metadata JSON -> SQLite DB")
        if db_path.exists() and args.rebuild:
            db_path.unlink()
        convert_metadata_to_db(
            metadata_json_path=str(metadata_json_path),
            db_path=str(db_path),
            dedup_by_title=False,
        )

    # 3) DB -> embedding_texts.json
    if embedding_texts_path.exists() and not args.rebuild:
        print(f"[Skip] embedding_texts exists: {embedding_texts_path}")
    else:
        print("\n[Step 3] Generating embedding_texts_v5.json")
        generate_embedding_texts_from_db(
            db_path=str(db_path),
            output_path=str(embedding_texts_path),
            fact_granularity=args.fact_granularity,
        )

    # 4) Dense path embeddings
    if embeddings_path.exists() and not args.rebuild:
        print(f"[Skip] embeddings exist: {embeddings_path}")
    else:
        print("\n[Step 4] Generating dense path embeddings")
        generator = PathEmbeddingGenerator(batch_size=int(args.embed_batch_size), max_concurrency=int(args.embed_concurrency))
        await generator.generate_embeddings(
            input_path=str(embedding_texts_path),
            output_path=str(embeddings_path),
        )

    # 5) BM25 index
    if (bm25_index_path / "metadata.json").exists() and not args.rebuild:
        print(f"[Skip] BM25 index exists: {bm25_index_path}")
    else:
        print("\n[Step 5] Building BM25 index")
        # bm25s saves multiple files; if rebuild, clear folder to avoid mixing.
        if bm25_index_path.exists() and args.rebuild:
            shutil.rmtree(bm25_index_path, ignore_errors=True)
        indexer = BM25Indexer(use_stemming=True)
        indexer.build_index(
            embedding_texts_path=str(embedding_texts_path),
            index_save_path=str(bm25_index_path),
            use_embedding_text_field=True,
            strip_stopwords_in_embedding_text=True,
        )

    manifest_path.write_text(json.dumps({
        'paper': '2026.emnlp-main.118', 'fact_granularity': args.fact_granularity,
        'dataset': cfg.name,
    }, indent=2), encoding='utf-8')
    print(f"\n[OK] Completed v5 pipeline for {cfg.name}")


async def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    if args.dataset == "all":
        for cfg in DATASETS.values():
            await build_for_dataset(args, cfg, root)
    else:
        cfg = DATASETS[str(args.dataset)]
        await build_for_dataset(args, cfg, root)


if __name__ == "__main__":
    asyncio.run(main())
