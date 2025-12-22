#!/usr/bin/env python3
"""Check doc_id uniqueness/coverage in corpus_idx metadata JSONs.

This answers: "Did we metadata-ize only unique doc_ids?" in the sense of
LLM calls/dedup and whether doc_id collisions produce inconsistent metadata.

Usage:
  python Analysis/check_unique_doc_ids_metadata.py \
    --paths MuSiQue/musique_sample_200_corpus_idx_metadata.json HotpotQA/hotpotqa_sample_200_corpus_idx_metadata.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple


def _stable_hash(obj: Any) -> str:
    b = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(b).hexdigest()


def _summarize(path: str) -> None:
    if not os.path.exists(path):
        print(f"== {path} ==")
        print("MISSING")
        return

    data = json.load(open(path, "r", encoding="utf-8"))

    doc_ids: List[str] = []
    missing_doc_id = 0
    missing_both_metadata_error = 0

    # Check consistency for repeated doc_ids (corpus_idx reused across multiple questions)
    hashes_by_doc_id: Dict[str, set] = {}

    for item in data:
        for cm in item.get("context_metadata", []) or []:
            did = cm.get("doc_id")
            if did is None or str(did).strip() == "":
                missing_doc_id += 1
                continue
            did_s = str(did)
            doc_ids.append(did_s)

            if ("metadata" not in cm) and ("error" not in cm):
                missing_both_metadata_error += 1

            # Only hash metadata when present
            if cm.get("metadata") is not None:
                hashes_by_doc_id.setdefault(did_s, set()).add(_stable_hash(cm.get("metadata")))

    counter = collections.Counter(doc_ids)
    unique_doc_ids = len(counter)
    dupe_doc_ids = sum(1 for _k, v in counter.items() if v > 1)

    inconsistent = sum(1 for _k, hs in hashes_by_doc_id.items() if len(hs) > 1)

    print(f"== {path} ==")
    print(f"context entries: {len(doc_ids)}")
    print(f"unique doc_ids:  {unique_doc_ids}")
    print(f"doc_ids reused across questions (count>1): {dupe_doc_ids}")
    print(f"missing doc_id entries: {missing_doc_id}")
    print(f"entries missing both 'metadata' and 'error' keys: {missing_both_metadata_error}")
    print(f"doc_ids with inconsistent metadata hashes: {inconsistent}")

    top_dupes = [x for x in counter.most_common(10) if x[1] > 1]
    if top_dupes:
        print("top reused doc_ids:")
        for did, n in top_dupes:
            print(f"  - {did}: {n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--paths",
        nargs="+",
        default=[
            "MuSiQue/musique_sample_200_corpus_idx_metadata.json",
            "HotpotQA/hotpotqa_sample_200_corpus_idx_metadata.json",
        ],
    )
    args = ap.parse_args()

    for p in args.paths:
        _summarize(p)
        print()


if __name__ == "__main__":
    main()
