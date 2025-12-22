#!/usr/bin/env python3
"""Prepare 200-sample files that preserve existing question IDs but add corpus_idx-aware formats.

Why:
- MuSiQue fixed corpus file assigns a global `corpus_idx` to each paragraph, so identical paragraphs
  across questions share the same ID.
- HotpotQA `hotpotqa_with_corpus_idx_by_title.json` assigns `corpus_idx` per title, enabling stable
  passage IDs across questions.

This script re-samples *the same 200 questions as existing sample files* by taking the ID list from:
- MuSiQue/musique_sample_200.json (field: _id)
- HotpotQA/hotpotqa_sample_200.json (field: _id)

And then filtering the corpus_idx-enriched sources:
- MuSiQue/musique_with_corpus_idx_fixed.json (field: id)
- HotpotQA/hotpotqa_with_corpus_idx_by_title.json (field: _id)

Outputs:
- MuSiQue/musique_sample_200_corpus_idx.json
- HotpotQA/hotpotqa_sample_200_corpus_idx.json

The output formats are *source-native*:
- MuSiQue keeps `paragraphs` (with is_supporting + corpus_idx) and also adds `_id` for consistency.
- HotpotQA keeps `context` as list[dict] with corpus_idx.

Usage:
  python prepare_corpus_idx_samples.py
  python prepare_corpus_idx_samples.py --out_suffix _corpus_idx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_id_list_from_sample(sample_path: Path, id_key: str) -> List[str]:
    data = _load_json(sample_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list JSON: {sample_path}")
    ids: List[str] = []
    for item in data:
        v = item.get(id_key)
        if v is None:
            raise ValueError(f"Missing {id_key} in sample item: {sample_path}")
        ids.append(str(v))
    return ids


def _index_by_key(items: List[dict], key: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for it in items:
        v = it.get(key)
        if v is None:
            continue
        out[str(v)] = it
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--musique_sample", default="MuSiQue/musique_sample_200.json")
    ap.add_argument("--musique_source", default="MuSiQue/musique_with_corpus_idx_fixed.json")
    ap.add_argument("--hotpot_sample", default="HotpotQA/hotpotqa_sample_200.json")
    ap.add_argument("--hotpot_source", default="HotpotQA/hotpotqa_with_corpus_idx_by_title.json")
    ap.add_argument("--out_suffix", default="_corpus_idx")
    args = ap.parse_args()

    # MuSiQue
    musique_sample_path = Path(args.musique_sample)
    musique_source_path = Path(args.musique_source)
    musique_ids = _get_id_list_from_sample(musique_sample_path, id_key="_id")

    musique_source = _load_json(musique_source_path)
    if not isinstance(musique_source, list):
        raise ValueError(f"Expected list JSON: {musique_source_path}")

    musique_by_id = _index_by_key(musique_source, key="id")

    musique_out: List[dict] = []
    missing_m = 0
    for qid in musique_ids:
        src = musique_by_id.get(qid)
        if not src:
            missing_m += 1
            continue
        item = dict(src)
        # For consistency with the rest of the repo
        item["_id"] = qid
        musique_out.append(item)

    out_musique = musique_sample_path.with_name(musique_sample_path.stem + args.out_suffix + ".json")
    _write_json(out_musique, musique_out)

    # HotpotQA
    hotpot_sample_path = Path(args.hotpot_sample)
    hotpot_source_path = Path(args.hotpot_source)
    hotpot_ids = _get_id_list_from_sample(hotpot_sample_path, id_key="_id")

    hotpot_source = _load_json(hotpot_source_path)
    if not isinstance(hotpot_source, list):
        raise ValueError(f"Expected list JSON: {hotpot_source_path}")

    hotpot_by_id = _index_by_key(hotpot_source, key="_id")

    hotpot_out: List[dict] = []
    missing_h = 0
    for qid in hotpot_ids:
        src = hotpot_by_id.get(qid)
        if not src:
            missing_h += 1
            continue
        hotpot_out.append(src)

    out_hotpot = hotpot_sample_path.with_name(hotpot_sample_path.stem + args.out_suffix + ".json")
    _write_json(out_hotpot, hotpot_out)

    print("=" * 80)
    print("[OK] Prepared corpus_idx-aware 200-samples")
    print(f"MuSiQue: {len(musique_out)}/{len(musique_ids)} -> {out_musique} (missing={missing_m})")
    print(f"HotpotQA: {len(hotpot_out)}/{len(hotpot_ids)} -> {out_hotpot} (missing={missing_h})")
    print("=" * 80)

    if missing_m or missing_h:
        print("[WARN] Some qids from old sample were not found in the corpus_idx sources.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
