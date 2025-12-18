#!/usr/bin/env python3
"""Debug CoT vs Non-CoT final@k retrieval (doc_id) differences.

This script compares two result JSON files (e.g., Non-CoT v11 vs CoT v12)
against a gold dataset, computing per-question recall@k and surfacing cases
where CoT is worse.

It is designed for *final@k* evaluation (k=5 by default) using doc_id mapping.
If the result file does not contain `final_retrieved_passages` / `final_retrieved_paths`,
we fall back to reconstructing final doc_ids from `decomposition.subquestions[].retrieved_paths`
using the same dedupe+score sort logic as the pipeline.

Usage (Hotpot):
  python Analysis/debug_compare_cot_vs_noncot_final_recall_at_k.py \
    --gold HotpotQA/hotpotqa_sample_200.json \
    --noncot Results/test_hotpot_v11_200_results_v4aligned.json \
    --cot Results/test_hotpot_v12_ragcot_results_v4aligned_v1.json

Usage (MuSiQue):
  python Analysis/debug_compare_cot_vs_noncot_final_recall_at_k.py \
    --gold MuSiQue/musique_sample_200.json \
    --noncot Results/test_musique_v11_200_results_v4aligned.json \
    --cot Results/test_musique_v12_ragcot_results_v4aligned_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure repo root is importable when running as a script from Analysis/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluate_retrieval import get_gold_doc_ids, get_final_retrieved_doc_ids_at_k


def _p(text: str = "") -> None:
    """Safe print for Windows consoles (e.g., cp949) that may not encode all unicode."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    data = (str(text) + "\n").encode(encoding, errors="backslashreplace")
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write(data)
    else:
        # Fallback: best-effort decode back to text
        sys.stdout.write(data.decode(encoding, errors="ignore"))


def _load_results(path: str) -> List[Dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON format: {path}")


def _load_gold(path: str) -> List[Dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data:
        # not used in this repo normally, but keep a safe fallback
        return data["data"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    raise ValueError(f"Unsupported gold JSON format: {path}")


def _gold_maps(gold: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    by_id = {item.get("_id"): item for item in gold if item.get("_id")}
    by_q = {item.get("question"): item for item in gold if item.get("question")}
    return by_id, by_q


def _find_gold_item(res_item: Dict, gold_by_id: Dict[str, Dict], gold_by_q: Dict[str, Dict]) -> Optional[Dict]:
    q = res_item.get("question")
    if q and q in gold_by_q:
        return gold_by_q[q]
    rid = res_item.get("id") or res_item.get("_id")
    if rid and rid in gold_by_id:
        return gold_by_id[rid]
    return None


@dataclass
class Row:
    qid: str
    question: str
    gold_size: int
    noncot_recall: float
    cot_recall: float
    noncot_doc_ids: List[str]
    cot_doc_ids: List[str]
    gold_doc_ids: List[str]


def _recall(gold: Set[str], retrieved: Set[str]) -> float:
    return (len(gold & retrieved) / len(gold)) if gold else 0.0


def _as_ordered_list_from_result(res_item: Dict, k: int) -> List[str]:
    # evaluate_retrieval returns a SET; we want ordered top-k for printing.
    # For deterministic printing, keep an order similar to the pipeline when possible.
    ordered: List[str] = []

    def push(d: str):
        if d and d not in ordered:
            ordered.append(d)

    # Prefer explicit final fields if present
    final_passages = res_item.get("final_retrieved_passages")
    if isinstance(final_passages, list):
        for p in final_passages:
            if isinstance(p, dict):
                push(str(p.get("doc_id") or ""))
            else:
                push(str(p))
            if len(ordered) >= k:
                return ordered

    final_paths = res_item.get("final_retrieved_paths")
    if isinstance(final_paths, list):
        for p in final_paths:
            if isinstance(p, dict):
                push(str(p.get("doc_id") or ""))
            else:
                push(str(p))
            if len(ordered) >= k:
                return ordered

    # Fallback: we only have a set from helper; sort for stable printing
    s = get_final_retrieved_doc_ids_at_k(res_item, k=k, sources="passages")
    return sorted(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--noncot", required=True)
    ap.add_argument("--cot", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=str, default="Analysis/debug_cot_vs_noncot_final_at_k_report.json")
    ap.add_argument("--top", type=int, default=30, help="How many worst gaps to print")
    args = ap.parse_args()

    gold = _load_gold(args.gold)
    gold_by_id, gold_by_q = _gold_maps(gold)

    noncot_results = _load_results(args.noncot)
    cot_results = _load_results(args.cot)

    noncot_by_qid = {r.get("id") or r.get("_id") or r.get("question"): r for r in noncot_results}
    cot_by_qid = {r.get("id") or r.get("_id") or r.get("question"): r for r in cot_results}

    common_keys = sorted(set(noncot_by_qid.keys()) & set(cot_by_qid.keys()))
    rows: List[Row] = []

    missing_gold = 0
    for key in common_keys:
        r_non = noncot_by_qid[key]
        r_cot = cot_by_qid[key]
        gold_item = _find_gold_item(r_non, gold_by_id, gold_by_q) or _find_gold_item(r_cot, gold_by_id, gold_by_q)
        if not gold_item:
            missing_gold += 1
            continue

        gold_set, _missing_titles, _ambig = get_gold_doc_ids(gold_item)
        non_set = get_final_retrieved_doc_ids_at_k(r_non, k=args.k, sources="passages")
        cot_set = get_final_retrieved_doc_ids_at_k(r_cot, k=args.k, sources="passages")

        rows.append(
            Row(
                qid=str(gold_item.get("_id") or key),
                question=str(gold_item.get("question") or r_non.get("question") or ""),
                gold_size=len(gold_set),
                noncot_recall=_recall(gold_set, non_set),
                cot_recall=_recall(gold_set, cot_set),
                noncot_doc_ids=_as_ordered_list_from_result(r_non, args.k),
                cot_doc_ids=_as_ordered_list_from_result(r_cot, args.k),
                gold_doc_ids=sorted(gold_set),
            )
        )

    worse = [r for r in rows if r.cot_recall < r.noncot_recall]
    equal = [r for r in rows if r.cot_recall == r.noncot_recall]
    better = [r for r in rows if r.cot_recall > r.noncot_recall]

    avg_non = sum(r.noncot_recall for r in rows) / max(1, len(rows))
    avg_cot = sum(r.cot_recall for r in rows) / max(1, len(rows))

    _p("=" * 100)
    _p(f"Gold:    {args.gold}")
    _p(f"Non-CoT: {args.noncot}")
    _p(f"CoT:     {args.cot}")
    _p(f"k={args.k} | common={len(rows)} | missing_gold={missing_gold}")
    _p(f"Avg recall@{args.k} noncot={avg_non:.4f} cot={avg_cot:.4f} (delta={avg_cot-avg_non:+.4f})")
    _p(f"Counts: worse={len(worse)} equal={len(equal)} better={len(better)}")
    _p("=" * 100)

    worst = sorted(worse, key=lambda r: (r.cot_recall - r.noncot_recall, r.gold_size))[: max(0, args.top)]
    for i, r in enumerate(worst, 1):
        _p(f"[{i:02d}] {r.qid} | noncot={r.noncot_recall:.2f} cot={r.cot_recall:.2f} | gold={r.gold_size}")
        _p(f"     Q: {r.question[:140]}")
        _p(f"     gold_doc_ids: {r.gold_doc_ids}")
        _p(f"     noncot@k:     {r.noncot_doc_ids}")
        _p(f"     cot@k:        {r.cot_doc_ids}")

    report = {
        "meta": {
            "gold": args.gold,
            "noncot": args.noncot,
            "cot": args.cot,
            "k": args.k,
            "common": len(rows),
            "missing_gold": missing_gold,
            "avg_recall_noncot": avg_non,
            "avg_recall_cot": avg_cot,
            "delta": avg_cot - avg_non,
            "counts": {"worse": len(worse), "equal": len(equal), "better": len(better)},
        },
        "worst_cases": [
            {
                "id": r.qid,
                "question": r.question,
                "gold_doc_ids": r.gold_doc_ids,
                "noncot_doc_ids_at_k": r.noncot_doc_ids,
                "cot_doc_ids_at_k": r.cot_doc_ids,
                "noncot_recall": r.noncot_recall,
                "cot_recall": r.cot_recall,
            }
            for r in worst
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _p(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
