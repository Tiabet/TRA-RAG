#!/usr/bin/env python3
"""Smoke tests for v5 (corpus_idx) artifact compatibility with the current retriever/pipeline.

What this checks:
- v5 artifacts exist (DB, embedding_texts, embeddings, BM25)
- HybridPathRetriever can load BM25 + dense and returns results
- (Optional) v11 pipeline can map retrieved doc_id -> original passage text when given the corpus_idx sample file

Usage:
  python Analysis/smoke_test_pipeline_v5_compat.py --dataset musique
  python Analysis/smoke_test_pipeline_v5_compat.py --dataset hotpot

Notes:
- Does NOT call any LLM.
"""

from __future__ import annotations

import argparse
import pathlib
import os
import sqlite3
import sys
from typing import Iterable

import asyncio

import numpy as np

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from hybrid_path_retriever import HybridPathRetriever


def _exists(path: str) -> bool:
    return os.path.exists(path)


def _require_files(base_dir: str, files: Iterable[str]) -> None:
    missing = [f for f in files if not _exists(os.path.join(base_dir, f))]
    if missing:
        raise SystemExit(f"Missing required files under {base_dir}: {missing}")


def _check_npz(npz_path: str) -> None:
    with np.load(npz_path, allow_pickle=True) as data:
        keys = sorted(list(data.keys()))
        print(f"NPZ keys: {keys}")
        if "embeddings" in data:
            print(f"embeddings shape/dtype: {data['embeddings'].shape} {data['embeddings'].dtype}")
        if "doc_ids" in data:
            doc_ids = data["doc_ids"]
            print(f"doc_ids shape/dtype: {doc_ids.shape} {doc_ids.dtype}")
            print(f"doc_ids sample: {list(doc_ids[:5])}")


def _check_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM metadata")
    n = cur.fetchone()[0]
    cur.execute("SELECT doc_id, source_title, entity_title FROM metadata LIMIT 3")
    sample = cur.fetchall()
    conn.close()
    print(f"DB rows: {n}")
    print("DB sample:")
    for row in sample:
        print(f"  - {row}")


async def _check_retriever(bm25_index: str, embeddings: str, query: str) -> None:
    r = HybridPathRetriever(
        bm25_index_path=bm25_index,
        embeddings_path=embeddings,
        bm25_weight=0.4,
        dense_weight=0.6,
    )
    out = await r.search_hybrid(query, top_k=5, bm25_candidates=50, dense_candidates=50)
    print(f"Retriever results: {len(out)}")
    for i, o in enumerate(out[:5]):
        print(
            f"  {i+1}. doc_id={o.get('doc_id')} source_title={o.get('source_title')} key_path={o.get('key_path')} value={o.get('value')}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["musique", "hotpot"], required=True)
    args = ap.parse_args()

    if args.dataset == "musique":
        base = "MuSiQue"
        db = f"{base}/metadata_v5.db"
        embedding_texts = f"{base}/embedding_texts_v5.json"
        embeddings = f"{base}/path_embeddings_v5.npz"
        bm25 = f"{base}/bm25_index_v5"
        query = "Beer Orders Wetherspoons"
    else:
        base = "HotpotQA"
        db = f"{base}/metadata_v5.db"
        embedding_texts = f"{base}/embedding_texts_v5.json"
        embeddings = f"{base}/path_embeddings_v5.npz"
        bm25 = f"{base}/bm25_index_v5"
        query = "Fountains of Wayne Welcome Interstate Managers"

    print("[v5 artifact existence]")
    for p in [db, embedding_texts, embeddings, bm25]:
        print(f"  - {p}: {'OK' if _exists(p) else 'MISSING'}")

    if not _exists(db):
        raise SystemExit("DB missing")
    if not _exists(embeddings):
        raise SystemExit("Embeddings missing")
    if not _exists(bm25):
        raise SystemExit("BM25 index missing")

    print("\n[DB check]")
    _check_db(db)

    print("\n[Dense embeddings (npz) check]")
    _check_npz(embeddings)

    print("\n[BM25 index files check]")
    _require_files(
        bm25,
        files=[
            "data.csc.index.npy",
            "indices.csc.index.npy",
            "indptr.csc.index.npy",
            "metadata.json",
            "params.index.json",
            "vocab.index.json",
        ],
    )
    print("BM25 index files: OK")

    print("\n[Retriever check]")
    asyncio.run(_check_retriever(bm25, embeddings, query))


if __name__ == "__main__":
    main()
