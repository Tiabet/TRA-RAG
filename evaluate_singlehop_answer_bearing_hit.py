#!/usr/bin/env python3
"""Evaluate whether top-k retrieval contains an answer-bearing passage.

An answer hit requires one gold answer alias to occur as a contiguous token
sequence in the retrieved passage. The primary metric includes the passage
title, matching the title-plus-body representation used by the retrievers.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Run:
    dataset: str
    method: str
    path: Path
    note: str = ""
    oracle: bool = False


RUNS = (
    Run("nq_rear", "Oracle", ROOT / "Results/nq_rear_upper.json", oracle=True),
    Run("nq_rear", "Cosine-Sim", ROOT / "Results/NaiveRAG/nq_rear_naive_noqd_cosine.json"),
    Run("nq_rear", "BM25", ROOT / "Results/NaiveRAG/nq_rear_naive_noqd_bm25.json"),
    Run("nq_rear", "Iter-RetGen", ROOT / "Results/NaiveRAG/nq_rear_naive_iter_iter_retgen_hybrid.json"),
    Run("nq_rear", "Self-Ask", ROOT / "Results/NaiveRAG/nq_rear_naive_iter_self_ask_hybrid.json"),
    Run("nq_rear", "ERARAG No-QD", ROOT / "Results/nq_rear_v12_noqd_repaired.json"),
    Run("nq_rear", "ERARAG QD", ROOT / "Results/nq_rear_v12_qd_repaired.json"),
    Run(
        "nq_rear",
        "ERARAG passages-only",
        ROOT / "Results/nq_rear_v12_noqd_repaired_passages_only.json",
    ),
    Run("popqa", "Oracle", ROOT / "Results/popqa_upper.json", oracle=True),
    Run("popqa", "Cosine-Sim", ROOT / "Results/NaiveRAG/popqa_naive_noqd_cosine.json"),
    Run("popqa", "BM25", ROOT / "Results/NaiveRAG/popqa_naive_noqd_bm25.json"),
    Run("popqa", "Iter-RetGen", ROOT / "Results/NaiveRAG/popqa_naive_iter_iter_retgen_hybrid.json"),
    Run("popqa", "Self-Ask", ROOT / "Results/NaiveRAG/popqa_naive_iter_self_ask_hybrid.json"),
    Run("nq_rear", "LinearRAG", ROOT / "Results/nq_rear_linearrag_qa.json"),
    Run("nq_rear", "NaiveRAG hybrid", ROOT / "Results/NaiveRAG/nq_rear_naive_noqd_hybrid.json"),
    Run(
        "nq_rear",
        "HippoRAG2",
        ROOT / "Results/nq_rear_hipporag2_retrieval_only.json",
        "Full-set retrieval-only rerun using the existing index.",
    ),
    Run("popqa", "ERARAG No-QD", ROOT / "Results/popqa_v12_noqd_canonical.json"),
    Run("popqa", "ERARAG QD", ROOT / "Results/popqa_v12_qd_canonical.json"),
    Run(
        "popqa",
        "ERARAG passages-only",
        ROOT / "Results/popqa_v12_noqd_canonical_passages_only.json",
    ),
    Run("popqa", "LinearRAG", ROOT / "Results/popqa_linearrag_qa_canonical.json"),
    Run("popqa", "SelfRAG", ROOT / "Results/popqa_selfrag_qa_canonical.json"),
    Run("popqa", "SelfRAG + reflection", ROOT / "Results/popqa_selfrag_reflect_qa.json"),
    Run("popqa", "NaiveRAG hybrid", ROOT / "Results/NaiveRAG/popqa_naive_noqd_hybrid.json"),
    Run(
        "popqa",
        "HippoRAG2",
        ROOT / "Results/popqa_hipporag2_retrieval_only.json",
        "Full-set canonical retrieval-only rerun using the existing index.",
    ),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def result_rows(path: Path) -> list[dict[str, Any]]:
    value = load_json(path)
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        return value["results"]
    raise ValueError(f"Unsupported result JSON shape: {path}")


def tokenize(value: Any) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).casefold()
    return tuple(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def contains_alias(tokens: tuple[str, ...], aliases: list[tuple[str, ...]]) -> bool:
    for alias in aliases:
        width = len(alias)
        if width and any(tokens[i : i + width] == alias for i in range(len(tokens) - width + 1)):
            return True
    return False


def corpus_path(dataset: str) -> Path:
    if dataset == "nq_rear":
        return ROOT / "NQ_REAR/nq_rear_corpus.json"
    if dataset == "popqa":
        return ROOT / "POPQA/popqa_corpus.json"
    raise ValueError(dataset)


def qa_path(dataset: str) -> Path:
    if dataset == "nq_rear":
        return ROOT / "NQ_REAR/nq_rear_qa.json"
    if dataset == "popqa":
        return ROOT / "POPQA/popqa_qa.json"
    raise ValueError(dataset)


def build_corpus(dataset: str) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    by_id: dict[str, tuple[str, str]] = {}
    by_row: dict[str, tuple[str, str]] = {}
    for row_index, row in enumerate(load_json(corpus_path(dataset))):
        title = str(row.get("title") or "")
        body = str(row.get("text") or row.get("paragraph_text") or "")
        passage = (title, body)
        by_row[str(row_index)] = passage
        by_id[str(row.get("idx", row_index))] = passage
    return by_id, by_row


def build_aliases(dataset: str) -> tuple[dict[str, list[tuple[str, ...]]], list[str]]:
    aliases_by_id: dict[str, list[tuple[str, ...]]] = {}
    order: list[str] = []
    for row in load_json(qa_path(dataset)):
        qid = str(row.get("id") or row.get("_id"))
        raw_aliases = [row.get("answer"), row.get("gold_answer")]
        value = row.get("answer_aliases")
        raw_aliases.extend(value if isinstance(value, list) else [value])
        seen: set[tuple[str, ...]] = set()
        aliases: list[tuple[str, ...]] = []
        for value in raw_aliases:
            alias = tokenize(value)
            if alias and alias not in seen:
                seen.add(alias)
                aliases.append(alias)
        aliases_by_id[qid] = aliases
        order.append(qid)
    return aliases_by_id, order


def build_supporting_ids(dataset: str) -> dict[str, list[str]]:
    return {
        str(row.get("id") or row.get("_id")): [str(x) for x in row.get("supporting_doc_ids", [])]
        for row in load_json(qa_path(dataset))
    }


def passage_text(
    passage: dict[str, Any],
    by_id: dict[str, tuple[str, str]],
    by_row: dict[str, tuple[str, str]],
) -> tuple[str, str, str]:
    title = str(passage.get("title") or "")
    body = str(passage.get("paragraph_text") or passage.get("body") or "")
    if passage.get("text") is not None:
        text = str(passage["text"])
        if not body and "\n" in text:
            parsed_title, body = text.split("\n", 1)
            title = title or parsed_title
        elif not body:
            body = text

    source = "inline"
    if not body:
        doc_id = passage.get("raw_doc_id", passage.get("doc_id"))
        stored = by_id.get(str(doc_id)) if doc_id is not None else None
        if stored is None and doc_id is not None:
            stored = by_row.get(str(doc_id))
            source = "corpus_row"
        else:
            source = "corpus_id"
        if stored is not None:
            stored_title, body = stored
            title = title or stored_title
        else:
            source = "unresolved"
    return title, body, source


def evaluate(run: Run, k: int) -> dict[str, Any]:
    rows = result_rows(run.path)
    aliases_by_id, qa_order = build_aliases(run.dataset)
    supporting_ids = build_supporting_ids(run.dataset) if run.oracle else {}
    by_id, by_row = build_corpus(run.dataset)
    title_body_hits = 0
    body_hits = 0
    missing_question = 0
    unresolved_passages = 0
    rows_with_fewer_than_k = 0
    id_order_mismatches = 0

    for index, row in enumerate(rows):
        qid = str(row.get("id") or row.get("_id") or row.get("question_id"))
        if index < len(qa_order) and qid != qa_order[index]:
            id_order_mismatches += 1
        aliases = aliases_by_id.get(qid)
        if aliases is None:
            missing_question += 1
            continue
        passages = (row.get("final_retrieved_passages") or [])[:k]
        if run.oracle and not passages:
            passages = [{"doc_id": doc_id} for doc_id in supporting_ids.get(qid, [])[:k]]
        if len(passages) < k:
            rows_with_fewer_than_k += 1
        hit_title_body = False
        hit_body = False
        for passage in passages:
            if not isinstance(passage, dict):
                passage = {"text": passage}
            title, body, source = passage_text(passage, by_id, by_row)
            unresolved_passages += int(source == "unresolved")
            hit_title_body = hit_title_body or contains_alias(tokenize(f"{title}\n{body}"), aliases)
            hit_body = hit_body or contains_alias(tokenize(body), aliases)
        title_body_hits += int(hit_title_body)
        body_hits += int(hit_body)

    evaluated = len(rows) - missing_question
    return {
        "dataset": run.dataset,
        "method": run.method,
        "path": str(run.path.relative_to(ROOT)),
        "rows": len(rows),
        "evaluated": evaluated,
        "answer_bearing_hit_at_k": title_body_hits / evaluated if evaluated else 0.0,
        "answer_bearing_body_only_hit_at_k": body_hits / evaluated if evaluated else 0.0,
        "title_only_increment": (title_body_hits - body_hits) / evaluated if evaluated else 0.0,
        "hits_title_body": title_body_hits,
        "hits_body_only": body_hits,
        "missing_question": missing_question,
        "id_order_mismatches": id_order_mismatches,
        "rows_with_fewer_than_k": rows_with_fewer_than_k,
        "unresolved_passages": unresolved_passages,
        "comparable_full_set": evaluated == 1000 and missing_question == 0,
        "note": run.note,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--at-k", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "Results/retrieval_eval/singlehop_answer_bearing_hit_at5.json",
    )
    args = parser.parse_args()

    results = []
    for run in RUNS:
        if not run.path.exists():
            results.append({"dataset": run.dataset, "method": run.method, "status": "missing"})
            continue
        result = evaluate(run, args.at_k)
        results.append(result)
        print(
            f"{result['dataset']}\t{result['method']}\tn={result['evaluated']}\t"
            f"title+body={100 * result['answer_bearing_hit_at_k']:.2f}\t"
            f"body={100 * result['answer_bearing_body_only_hit_at_k']:.2f}\t"
            f"id_mismatch={result['id_order_mismatches']}\t"
            f"unresolved={result['unresolved_passages']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric": "Answer-bearing Hit@k",
        "at_k": args.at_k,
        "primary_passage_representation": "title + body",
        "matching": "case-folded NFKC alphanumeric contiguous-token match against any gold answer alias",
        "results": results,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
