#!/usr/bin/env python3
"""Analyze relation target list "explosion" in embedding_texts_v5.json.

We previously changed EmbeddingTextGenerator so that if a relation's target is a list,
we emit one embedding entry per target element.

This script estimates the impact by looking for repeated (doc_id, key_path) pairs
among relation-like entries (value contains a JSON object with a 'target' field).

Run:
  .venv/Scripts/python.exe Analysis/analyze_relation_list_explosion.py --limit 200000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple


def stream_json_array(path: Path, limit: Optional[int] = None, chunk_size: int = 1 << 16) -> Iterator[Dict]:
    """Stream a top-level JSON array without loading it entirely."""
    dec = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""

        # Seek to the start of the array.
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"EOF before array start in {path}")
            buf += chunk
            i = buf.find("[")
            if i != -1:
                buf = buf[i + 1 :]
                break

        n = 0
        while True:
            # Skip whitespace and commas.
            j = 0
            while j < len(buf) and buf[j] in " \t\r\n,":
                j += 1
            buf = buf[j:]

            if not buf:
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                buf += chunk
                continue

            if buf[0] == "]":
                return

            # Ensure we can decode the next object.
            while True:
                try:
                    obj, idx = dec.raw_decode(buf)
                    break
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buf += chunk

            buf = buf[idx:]
            if isinstance(obj, dict):
                yield obj
                n += 1
                if limit is not None and n >= limit:
                    return


def is_relation_like(entry: Dict) -> bool:
    value = entry.get("value")
    # In our embedding texts, relation entries typically store value as a JSON string like {"target": ...}
    return isinstance(value, str) and "\"target\"" in value and value.lstrip().startswith("{")


def analyze_file(path: Path, limit: Optional[int]) -> Dict:
    total = 0
    relation_total = 0

    # Multiplicity of relation facts: same (doc_id, key_path) appearing many times.
    rel_doc_key_counts: Counter[Tuple[str, str]] = Counter()
    rel_doc_counts: Counter[str] = Counter()
    rel_key_counts: Counter[str] = Counter()

    for entry in stream_json_array(path, limit=limit):
        total += 1
        if not is_relation_like(entry):
            continue

        relation_total += 1
        doc_id = str(entry.get("doc_id") or "")
        key_path = str(entry.get("key_path") or "")
        rel_doc_key_counts[(doc_id, key_path)] += 1
        rel_doc_counts[doc_id] += 1
        rel_key_counts[key_path] += 1

    multi_doc_key = sum(1 for _k, c in rel_doc_key_counts.items() if c > 1)
    heavy_doc_key = sum(1 for _k, c in rel_doc_key_counts.items() if c >= 5)

    return {
        "file": str(path).replace("\\", "/"),
        "scanned": total,
        "relation_like": relation_total,
        "relation_like_ratio": (relation_total / total if total else 0.0),
        "unique_relation_doc_keys": len(rel_doc_key_counts),
        "multi_relation_doc_keys": multi_doc_key,
        "multi_relation_doc_key_ratio": (multi_doc_key / len(rel_doc_key_counts) if rel_doc_key_counts else 0.0),
        "heavy_relation_doc_keys_ge5": heavy_doc_key,
        "top_relation_keys": rel_key_counts.most_common(10),
        "top_docs_by_relation_count": rel_doc_counts.most_common(10),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit",
        type=int,
        default=200000,
        help="Max entries to scan per file (default: 200000). Use 0 for full scan.",
    )
    args = ap.parse_args()
    limit = None if args.limit == 0 else args.limit

    files = [
        Path("MuSiQue/embedding_texts_v5.json"),
        Path("HotpotQA/embedding_texts_v5.json"),
    ]

    for p in files:
        if not p.exists():
            print(f"[SKIP] Missing: {p}")
            continue
        stats = analyze_file(p, limit=limit)
        print("=" * 100)
        for k, v in stats.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
