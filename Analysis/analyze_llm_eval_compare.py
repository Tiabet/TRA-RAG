#!/usr/bin/env python3
"""Compare LLM-evaluation outputs between two runs (e.g., v4aligned vs v5).

Produces:
- Summary deltas (accuracy, incorrect count)
- Error type breakdown (Insufficient, partial, entity mismatch, numeric/date-like, other)
- Questions where verdict flipped (CORRECT->INCORRECT / INCORRECT->CORRECT)
- Sanity checks: question set alignment and duplicates

Usage:
  .venv/Scripts/python.exe Analysis/analyze_llm_eval_compare.py \
    --a Results/llm_eval/llm_eval_test_musique_v11_200_results_v4aligned.json \
    --b Results/llm_eval/llm_eval_test_musique_v11_200_results_v5.json \
    --show 10
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvalItem:
    question: str
    gold: str
    pred: str
    verdict: str
    confidence: str
    reason: str


def load_eval(path: Path) -> List[EvalItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", []) if isinstance(data, dict) else []
    out: List[EvalItem] = []
    for r in results:
        q = str(r.get("question") or "").strip()
        g = str(r.get("gold_answer") or "").strip()
        p = str(r.get("predicted_answer") or "").strip()
        ev = r.get("evaluation") or {}
        verdict = str(ev.get("verdict") or "").strip()
        conf = str(ev.get("confidence") or "").strip()
        reason = str(ev.get("reason") or "").strip()
        if q:
            out.append(EvalItem(q, g, p, verdict, conf, reason))
    return out


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip())


def classify_error(item: EvalItem) -> str:
    pred = item.pred.lower()
    reason = item.reason.lower()

    if "insufficient information" in pred:
        return "insufficient"

    # Heuristics based on evaluator reason text
    if any(k in reason for k in ["different", "does not match", "mismatch", "incorrect", "wrong"]):
        # Detect numeric/date-like mismatches
        if re.search(r"\b(19\d{2}|20\d{2})\b", item.pred) or re.search(r"\b(19\d{2}|20\d{2})\b", item.gold):
            return "year/date"
        if re.search(r"\b\d+\b", item.pred) and re.search(r"\b\d+\b", item.gold):
            return "numeric"
        # Entity-ish mismatch: proper nouns often capitalized
        if re.search(r"[A-Z][a-z]+", item.pred) and re.search(r"[A-Z][a-z]+", item.gold):
            return "entity"
        return "other_mismatch"

    # Fall back to comparing normalized strings
    if item.pred.strip() and item.gold.strip() and item.pred.strip() != item.gold.strip():
        if re.search(r"\b\d+\b", item.pred) and re.search(r"\b\d+\b", item.gold):
            return "numeric"
        return "other"

    return "other"


def verdict_counts(items: List[EvalItem]) -> Counter:
    c = Counter()
    for it in items:
        c[it.verdict] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Path to llm_eval json A")
    ap.add_argument("--b", required=True, help="Path to llm_eval json B")
    ap.add_argument("--show", type=int, default=10, help="How many flip examples to print")
    args = ap.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)

    a = load_eval(a_path)
    b = load_eval(b_path)

    a_map = {norm_q(x.question): x for x in a}
    b_map = {norm_q(x.question): x for x in b}

    a_qs = list(a_map.keys())
    b_qs = list(b_map.keys())

    # Alignment checks
    a_dups = len(a) - len(a_map)
    b_dups = len(b) - len(b_map)

    only_a = sorted(set(a_map) - set(b_map))
    only_b = sorted(set(b_map) - set(a_map))

    print("=" * 110)
    print(f"A: {a_path.as_posix()}  items={len(a)} unique_q={len(a_map)} dups={a_dups}")
    print(f"B: {b_path.as_posix()}  items={len(b)} unique_q={len(b_map)} dups={b_dups}")
    print(f"Question-set diff: only_A={len(only_a)} only_B={len(only_b)}")

    # Basic accuracy from verdict
    a_counts = verdict_counts(a)
    b_counts = verdict_counts(b)
    a_total = sum(a_counts.values())
    b_total = sum(b_counts.values())
    a_acc = (a_counts.get("CORRECT", 0) / a_total) if a_total else 0.0
    b_acc = (b_counts.get("CORRECT", 0) / b_total) if b_total else 0.0

    print("-" * 110)
    print(f"A verdicts: {dict(a_counts)}  accuracy={a_acc:.4f}")
    print(f"B verdicts: {dict(b_counts)}  accuracy={b_acc:.4f}")
    print(f"Delta accuracy (B-A): {b_acc - a_acc:+.4f}")

    # Error type breakdown (INCORRECT only)
    a_err = [x for x in a if x.verdict == "INCORRECT"]
    b_err = [x for x in b if x.verdict == "INCORRECT"]

    a_types = Counter(classify_error(x) for x in a_err)
    b_types = Counter(classify_error(x) for x in b_err)

    print("-" * 110)
    print("Error-type breakdown (INCORRECT only):")
    all_keys = sorted(set(a_types) | set(b_types))
    for k in all_keys:
        print(f"  {k:14s}  A={a_types.get(k,0):3d}  B={b_types.get(k,0):3d}  delta={b_types.get(k,0)-a_types.get(k,0):+3d}")

    # Verdict flips
    flips_ca: List[Tuple[str, EvalItem, EvalItem]] = []  # correct->incorrect
    flips_ac: List[Tuple[str, EvalItem, EvalItem]] = []  # incorrect->correct

    for q in set(a_map) & set(b_map):
        ai = a_map[q]
        bi = b_map[q]
        if ai.verdict == "CORRECT" and bi.verdict == "INCORRECT":
            flips_ca.append((q, ai, bi))
        elif ai.verdict == "INCORRECT" and bi.verdict == "CORRECT":
            flips_ac.append((q, ai, bi))

    print("-" * 110)
    print(f"Flips CORRECT->INCORRECT: {len(flips_ca)}")
    print(f"Flips INCORRECT->CORRECT: {len(flips_ac)}")

    show_n = max(0, int(args.show))
    if show_n:
        def _print_examples(title: str, flips: List[Tuple[str, EvalItem, EvalItem]]):
            print("-" * 110)
            print(title)
            for q, ai, bi in flips[:show_n]:
                print(f"Q: {ai.question}")
                print(f"  Gold: {ai.gold}")
                print(f"  A({ai.verdict}): {ai.pred}")
                print(f"  B({bi.verdict}): {bi.pred}")
                print(f"  B_reason: {bi.reason[:160]}")
                print("")

        _print_examples("Examples CORRECT->INCORRECT", flips_ca)
        _print_examples("Examples INCORRECT->CORRECT", flips_ac)


if __name__ == "__main__":
    main()
