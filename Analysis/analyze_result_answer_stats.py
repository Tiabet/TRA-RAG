#!/usr/bin/env python3
"""Compare answer statistics across result files.

Focus:
- 빈 답/None/Insufficient information 비율
- 답변 길이(단어/문자) 분포

Usage:
  .venv/Scripts/python.exe Analysis/analyze_result_answer_stats.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return data["results"]
    if isinstance(data, list):
        return data
    return []


def get_pred(item: Dict[str, Any]) -> Any:
    if "predicted_answer" in item:
        return item.get("predicted_answer")
    if "final_answer" in item:
        return item.get("final_answer")
    if "prediction" in item:
        return item.get("prediction")
    return None


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    null_pred = 0
    empty_pred = 0
    insufficient = 0

    # Final retrieval stats
    num_passages_vals: List[int] = []
    num_paths_vals: List[int] = []
    final_passage_unique_counts: List[int] = []
    final_passage_total_counts: List[int] = []
    final_passage_dup_ratio_vals: List[float] = []
    final_passage_missing_doc_id = 0

    word_counts: List[int] = []
    char_counts: List[int] = []
    over_5_words = 0

    for r in results:
        # Final retrieval stats (best-effort; field may be absent in older results)
        if isinstance(r.get("num_passages"), int):
            num_passages_vals.append(int(r.get("num_passages")))
        if isinstance(r.get("num_paths"), int):
            num_paths_vals.append(int(r.get("num_paths")))

        frp = r.get("final_retrieved_passages")
        if isinstance(frp, list):
            final_passage_total_counts.append(len(frp))
            doc_ids = []
            for p in frp:
                if isinstance(p, dict):
                    did = p.get("doc_id")
                    if did is None or str(did).strip() == "":
                        final_passage_missing_doc_id += 1
                    else:
                        doc_ids.append(str(did))
            uniq = len(set(doc_ids))
            total = len(doc_ids)
            final_passage_unique_counts.append(uniq)
            if total > 0:
                final_passage_dup_ratio_vals.append(1.0 - (uniq / total))

        pred = get_pred(r)
        if pred is None:
            null_pred += 1
            continue
        pred_s = str(pred).strip()
        if pred_s == "":
            empty_pred += 1
            continue
        if "Insufficient information" in pred_s:
            insufficient += 1

        wc = len([w for w in re.split(r"\s+", pred_s) if w])
        word_counts.append(wc)
        char_counts.append(len(pred_s))
        if wc > 5:
            over_5_words += 1

    def avg(xs: List[int]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    def q(xs: List[int], p: float) -> int:
        if not xs:
            return 0
        xs2 = sorted(xs)
        idx = int(round((len(xs2) - 1) * p))
        return xs2[idx]

    def avgf(xs: List[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    denom = max(1, len(word_counts))

    return {
        "n": n,
        "null_pred": null_pred,
        "empty_pred": empty_pred,
        "insufficient": insufficient,
        "pct_null": null_pred / max(1, n),
        "pct_empty": empty_pred / max(1, n),
        "pct_insufficient": insufficient / max(1, n),
        "avg_words": avg(word_counts),
        "p50_words": q(word_counts, 0.5),
        "p90_words": q(word_counts, 0.9),
        "pct_over_5_words": over_5_words / denom,
        "avg_chars": avg(char_counts),
        "p50_chars": q(char_counts, 0.5),
        "p90_chars": q(char_counts, 0.9),

        # Final retrieval stats
        "avg_num_passages_field": avg(num_passages_vals),
        "p10_num_passages_field": q(num_passages_vals, 0.1),
        "p50_num_passages_field": q(num_passages_vals, 0.5),
        "p90_num_passages_field": q(num_passages_vals, 0.9),
        "avg_num_paths_field": avg(num_paths_vals),
        "avg_final_passages_total": avg(final_passage_total_counts),
        "avg_final_passages_unique_doc_ids": avg(final_passage_unique_counts),
        "avg_final_passage_dup_ratio": avgf(final_passage_dup_ratio_vals),
        "final_passage_missing_doc_id": final_passage_missing_doc_id,
    }


def main() -> None:
    pairs: List[Tuple[str, str]] = [
        ("MuSiQue v4aligned", "Results/test_musique_v11_200_results_v4aligned.json"),
        ("MuSiQue v5", "Results/test_musique_v11_200_results_v5.json"),
        ("Hotpot v4aligned", "Results/test_hotpot_v11_200_results_v4aligned.json"),
        ("Hotpot v5", "Results/test_hotpot_v11_200_results_v5.json"),
    ]

    for name, rel in pairs:
        p = Path(rel)
        if not p.exists():
            print(f"[MISSING] {name}: {rel}")
            continue
        res = load_results(p)
        s = summarize(res)
        print("\n" + "=" * 90)
        print(f"{name}: {rel}")
        for k in [
            "n",
            "pct_insufficient",
            "pct_empty",
            "pct_null",
            "avg_words",
            "p50_words",
            "p90_words",
            "pct_over_5_words",
            "avg_chars",
            "p90_chars",
            "avg_num_passages_field",
            "p10_num_passages_field",
            "p50_num_passages_field",
            "p90_num_passages_field",
            "avg_final_passages_unique_doc_ids",
            "avg_final_passage_dup_ratio",
        ]:
            print(f"{k}: {s[k]}")


if __name__ == "__main__":
    main()
