#!/usr/bin/env python3
"""Prepare HippoRAG2 single-hop datasets for the v5 pipeline.

Creates two files per dataset:
- *_qa.json: compact QA records used by runners.
- *_corpus_for_pipeline.json: one QA-like wrapper over the full corpus used for indexing.

The wrapper matches the existing MuSiQue-style parser:
items[].paragraphs[] = {title, paragraph_text, corpus_idx, local_idx, is_supporting}
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SingleHopPaths:
    name: str
    raw_qa: Path
    raw_corpus: Path
    qa_out: Path
    corpus_out: Path


def _norm_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()]
            if parsed is not None:
                return [str(parsed)]
        except Exception:
            return [s]
    return [str(value)]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_corpus_wrapper(dataset_name: str, raw_corpus: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
    paragraphs: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], str] = {}

    for i, doc in enumerate(raw_corpus):
        if not isinstance(doc, dict):
            continue
        raw_idx = doc.get("idx", i)
        doc_id = str(raw_idx)
        title = _norm_text(doc.get("title"))
        text = _norm_text(doc.get("text"))
        if not title and not text:
            continue
        paragraphs.append(
            {
                "idx": i,
                "local_idx": i,
                "corpus_idx": doc_id,
                "title": title or f"doc_{doc_id}",
                "paragraph_text": text,
                "is_supporting": False,
            }
        )
        lookup.setdefault((title, text), doc_id)

    wrapper = [
        {
            "id": f"{dataset_name}_corpus",
            "question": "",
            "answer": "",
            "answer_aliases": [],
            "paragraphs": paragraphs,
        }
    ]
    return wrapper, lookup


def _supporting_from_contexts(contexts: Any, lookup: dict[tuple[str, str], str]) -> tuple[list[str], list[str]]:
    doc_ids: list[str] = []
    titles: list[str] = []
    if not isinstance(contexts, list):
        return doc_ids, titles

    seen: set[str] = set()
    for ctx in contexts:
        if not isinstance(ctx, dict) or not ctx.get("is_supporting"):
            continue
        title = _norm_text(ctx.get("title"))
        text = _norm_text(ctx.get("text") or ctx.get("paragraph_text"))
        doc_id = lookup.get((title, text))
        if doc_id is None:
            continue
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
        titles.append(title)
    return doc_ids, titles


def prepare_nq_rear(paths: SingleHopPaths) -> dict[str, int]:
    raw_qa = _load_json(paths.raw_qa)
    raw_corpus = _load_json(paths.raw_corpus)
    corpus_wrapper, lookup = _build_corpus_wrapper(paths.name, raw_corpus)

    qa_items: list[dict[str, Any]] = []
    for i, item in enumerate(raw_qa):
        aliases = _safe_json_list(item.get("reference"))
        answer = aliases[0] if aliases else ""
        supporting_doc_ids, supporting_titles = _supporting_from_contexts(item.get("contexts"), lookup)
        qa_items.append(
            {
                "id": f"nq_rear_{i}",
                "_id": f"nq_rear_{i}",
                "question": _norm_text(item.get("question")),
                "answer": answer,
                "gold_answer": answer,
                "answer_aliases": aliases,
                "supporting_doc_ids": supporting_doc_ids,
                "supporting_titles": supporting_titles,
                "source": "nq_rear",
            }
        )

    _write_json(paths.qa_out, qa_items)
    _write_json(paths.corpus_out, corpus_wrapper)
    return {
        "qa_items": len(qa_items),
        "corpus_docs": len(corpus_wrapper[0]["paragraphs"]),
        "supporting_doc_ids": sum(len(x.get("supporting_doc_ids", [])) for x in qa_items),
    }


def prepare_popqa(paths: SingleHopPaths) -> dict[str, int]:
    raw_qa = _load_json(paths.raw_qa)
    raw_corpus = _load_json(paths.raw_corpus)
    corpus_wrapper, lookup = _build_corpus_wrapper(paths.name, raw_corpus)

    qa_items: list[dict[str, Any]] = []
    for i, item in enumerate(raw_qa):
        aliases = _safe_json_list(item.get("possible_answers"))
        answer = aliases[0] if aliases else _norm_text(item.get("obj"))
        if answer and answer not in aliases:
            aliases.insert(0, answer)
        qid = _norm_text(item.get("id")) or f"popqa_{i}"
        supporting_doc_ids, supporting_titles = _supporting_from_contexts(item.get("paragraphs"), lookup)
        qa_items.append(
            {
                "id": qid,
                "_id": qid,
                "question": _norm_text(item.get("question")),
                "answer": answer,
                "gold_answer": answer,
                "answer_aliases": aliases,
                "supporting_doc_ids": supporting_doc_ids,
                "supporting_titles": supporting_titles,
                "subj": item.get("subj"),
                "prop": item.get("prop"),
                "obj": item.get("obj"),
                "source": "popqa",
            }
        )

    _write_json(paths.qa_out, qa_items)
    _write_json(paths.corpus_out, corpus_wrapper)
    return {
        "qa_items": len(qa_items),
        "corpus_docs": len(corpus_wrapper[0]["paragraphs"]),
        "supporting_doc_ids": sum(len(x.get("supporting_doc_ids", [])) for x in qa_items),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare NQ_REAR/POPQA for the v5 pipeline")
    p.add_argument("--dataset", choices=["all", "nq_rear", "popqa"], default="all")
    p.add_argument("--root", type=str, default=".")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    configs = {
        "nq_rear": SingleHopPaths(
            name="nq_rear",
            raw_qa=root / "NQ_REAR" / "nq_rear.json",
            raw_corpus=root / "NQ_REAR" / "nq_rear_corpus.json",
            qa_out=root / "NQ_REAR" / "nq_rear_qa.json",
            corpus_out=root / "NQ_REAR" / "nq_rear_corpus_for_pipeline.json",
        ),
        "popqa": SingleHopPaths(
            name="popqa",
            raw_qa=root / "POPQA" / "popqa.json",
            raw_corpus=root / "POPQA" / "popqa_corpus.json",
            qa_out=root / "POPQA" / "popqa_qa.json",
            corpus_out=root / "POPQA" / "popqa_corpus_for_pipeline.json",
        ),
    }

    selected = list(configs) if args.dataset == "all" else [str(args.dataset)]
    for name in selected:
        cfg = configs[name]
        if not cfg.raw_qa.exists():
            raise FileNotFoundError(cfg.raw_qa)
        if not cfg.raw_corpus.exists():
            raise FileNotFoundError(cfg.raw_corpus)
        stats = prepare_nq_rear(cfg) if name == "nq_rear" else prepare_popqa(cfg)
        print(f"[OK] {name}: {stats}")
        print(f"  qa:     {cfg.qa_out}")
        print(f"  corpus: {cfg.corpus_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
