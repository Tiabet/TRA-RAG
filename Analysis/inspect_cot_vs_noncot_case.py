#!/usr/bin/env python3
"""Inspect one question deeply: CoT vs Non-CoT retrieval + SQ answers.

This prints:
- gold doc_ids
- final@k doc_ids for each run
- per-subquestion question/answer and the *actual_question* used for retrieval
  (uses stored `actual_question` if present; otherwise reconstructs from placeholders)
- top ranked UNIQUE paths (doc_id + score + key_path/value) that drive final selection

Usage:
  python Analysis/inspect_cot_vs_noncot_case.py \
    --gold HotpotQA/hotpotqa_sample_200.json \
    --noncot Results/test_hotpot_v11_200_results_v4aligned.json \
    --cot Results/test_hotpot_v12_ragcot_results_v4aligned_v1.json \
    --qid 5a74524455429979e28828f6

Or by question substring:
  python Analysis/inspect_cot_vs_noncot_case.py ... --question_contains "Hwarang"
"""

from __future__ import annotations

import argparse
import json
import sys
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
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    raise ValueError(f"Unsupported gold JSON format: {path}")


def _gold_maps(gold: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    by_id = {item.get("_id"): item for item in gold if item.get("_id")}
    by_q = {item.get("question"): item for item in gold if item.get("question")}
    return by_id, by_q


def _find_result_by_qid_or_question(results: List[Dict], qid: Optional[str], question_contains: Optional[str]) -> Dict:
    if qid:
        for r in results:
            rid = r.get("id") or r.get("_id")
            if rid and str(rid) == str(qid):
                return r
    if question_contains:
        qlc = question_contains.lower()
        for r in results:
            q = (r.get("question") or "").lower()
            if qlc in q:
                return r
    raise SystemExit("Could not find matching result. Provide --qid or --question_contains.")


def _find_gold_item(res_item: Dict, gold_by_id: Dict[str, Dict], gold_by_q: Dict[str, Dict]) -> Optional[Dict]:
    q = res_item.get("question")
    if q and q in gold_by_q:
        return gold_by_q[q]
    rid = res_item.get("id") or res_item.get("_id")
    if rid and rid in gold_by_id:
        return gold_by_id[rid]
    return None


def _path_dedupe_key(p: Dict) -> Tuple[str, str, str, str]:
    source_title = p.get('source_title') or p.get('title') or ''
    entity_title = p.get('entity_title') or p.get('title') or ''
    key_path = p.get('key_path', '')
    value = p.get('value', '')
    return (str(source_title), str(entity_title), str(key_path), str(value))


def _safe_score(p: Dict) -> float:
    try:
        s = p.get('score', None)
        return float(s) if s is not None else float('-inf')
    except Exception:
        return float('-inf')


def _collect_unique_paths_sorted(res_item: Dict) -> List[Dict]:
    d = res_item.get('decomposition') or {}
    subqs = []
    if isinstance(d, dict):
        subqs = d.get('subquestions', []) or []
    elif isinstance(d, list):
        subqs = d

    seen = set()
    uniq: List[Dict] = []
    for sq in subqs:
        for p in (sq.get('retrieved_paths') or []):
            if not isinstance(p, dict):
                continue
            k = _path_dedupe_key(p)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(p)

    return sorted(uniq, key=_safe_score, reverse=True)


def _reconstruct_actual_question(sq: Dict, all_subqs: List[Dict]) -> str:
    # If pipeline stored it, trust it.
    if sq.get('actual_question'):
        return str(sq.get('actual_question'))

    q = str(sq.get('question') or '')

    # Replace [SQ{N}_Answer]
    for dep in all_subqs:
        dep_id = dep.get('id')
        ans = dep.get('answer')
        if not dep_id or not ans:
            continue
        q = q.replace(f"[{dep_id}_Answer]", str(ans))
        # Replace MuSiQue '#N'
        if str(dep_id).startswith('SQ'):
            num = str(dep_id)[2:]
            if num:
                q = q.replace(f"#{num}", str(ans))

    return q


def _print_subqs(res_item: Dict, label: str):
    d = res_item.get('decomposition') or {}
    subqs = []
    if isinstance(d, dict):
        subqs = d.get('subquestions', []) or []
    elif isinstance(d, list):
        subqs = d

    _p(f"\n[{label}] subquestions={len(subqs)}")
    for sq in subqs:
        sid = sq.get('id')
        qq = sq.get('question')
        aa = sq.get('answer')
        actual = _reconstruct_actual_question(sq, subqs)
        _p(f"- {sid}: {str(qq)[:140]}")
        _p(f"  answer: {str(aa)[:200]}")
        _p(f"  actual_question: {str(actual)[:220]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--noncot', required=True)
    ap.add_argument('--cot', required=True)
    ap.add_argument('--qid', type=str, default=None)
    ap.add_argument('--question_contains', type=str, default=None)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--top_paths', type=int, default=12)
    args = ap.parse_args()

    gold = _load_gold(args.gold)
    gold_by_id, gold_by_q = _gold_maps(gold)

    noncot_results = _load_results(args.noncot)
    cot_results = _load_results(args.cot)

    r_non = _find_result_by_qid_or_question(noncot_results, args.qid, args.question_contains)
    r_cot = _find_result_by_qid_or_question(cot_results, args.qid, args.question_contains)

    gold_item = _find_gold_item(r_non, gold_by_id, gold_by_q) or _find_gold_item(r_cot, gold_by_id, gold_by_q)
    if not gold_item:
        raise SystemExit("Could not map to gold item by id or question.")

    gold_set, missing_titles, ambig_titles = get_gold_doc_ids(gold_item)
    non_set = get_final_retrieved_doc_ids_at_k(r_non, k=args.k, sources='passages')
    cot_set = get_final_retrieved_doc_ids_at_k(r_cot, k=args.k, sources='passages')

    _p('=' * 100)
    _p(f"ID: {gold_item.get('_id')}")
    _p(f"Q:  {gold_item.get('question')}")
    _p(f"Gold doc_ids: {sorted(gold_set)}")
    if missing_titles or ambig_titles:
        _p(f"Gold mapping: missing_titles={len(missing_titles)} ambiguous_titles={len(ambig_titles)}")
    _p('-' * 100)
    _p(f"Non-CoT final@{args.k}: {sorted(non_set)}")
    _p(f"CoT    final@{args.k}: {sorted(cot_set)}")
    _p(f"Non-CoT misses: {sorted(gold_set - non_set)}")
    _p(f"CoT    misses: {sorted(gold_set - cot_set)}")

    _print_subqs(r_non, 'Non-CoT')
    _print_subqs(r_cot, 'CoT')

    _p('\n[Top UNIQUE paths driving final selection]')
    non_paths = _collect_unique_paths_sorted(r_non)[: max(0, args.top_paths)]
    cot_paths = _collect_unique_paths_sorted(r_cot)[: max(0, args.top_paths)]

    def show(paths: List[Dict], label: str):
        _p(f"\n{label} (top {len(paths)})")
        for i, p in enumerate(paths, 1):
            doc_id = p.get('doc_id')
            score = p.get('score')
            key_path = (p.get('key_path') or '')
            value = p.get('value')
            if isinstance(value, str) and len(value) > 120:
                value = value[:120] + '...'
            _p(f"[{i:02d}] doc_id={doc_id} score={score} key_path={key_path} value={value}")

    show(non_paths, 'Non-CoT')
    show(cot_paths, 'CoT')

    _p('=' * 100)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
