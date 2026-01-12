#!/usr/bin/env python3
"""Prepare LVEVAL inputs for ChunkRAG v5 pipeline.

LVEVAL provides:
- LVEVAL/lveval_corpus.json: global corpus list[{idx,title,text}, ...]
- LVEVAL/lveval.json: QA list[dict] where `context` is a huge string with '### Passage N' blocks

Our v5 indexing pipeline expects a QA-like JSON where passages are provided via
`paragraphs` or `context` with stable `corpus_idx`.

This module generates:
- LVEVAL/lveval_corpus_for_pipeline.json: QA-like wrapper over the corpus, one doc per item.
  This is used ONLY for building metadata/DB/embedding_texts/indices.
- LVEVAL/lveval_qa_compact.json: QA file without the giant `context` string.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_corpus_wrapper_for_pipeline(*, corpus_path: Path, out_path: Path) -> None:
    corpus = _load_json(corpus_path)
    if not isinstance(corpus, list):
        raise ValueError(f"Expected list corpus JSON, got {type(corpus).__name__}: {corpus_path}")

    items: list[dict[str, Any]] = []
    for i, doc in enumerate(corpus):
        if not isinstance(doc, dict):
            raise ValueError(f"corpus[{i}] not a dict")

        idx = doc.get("idx")
        if idx is None:
            raise ValueError(f"corpus[{i}] missing idx")
        try:
            corpus_idx = int(idx)
        except Exception as e:
            raise ValueError(f"corpus[{i}] idx not int-coercible: {idx!r}") from e

        title = str(doc.get("title") or "")
        text = str(doc.get("text") or "")

        # One document per item. This avoids duplicating documents across QA questions.
        items.append(
            {
                "id": str(corpus_idx),
                "question": "",
                "answer": "",
                "paragraphs": [
                    {
                        "title": title,
                        "paragraph_text": text,
                        "corpus_idx": corpus_idx,
                        "local_idx": 0,
                    }
                ],
            }
        )

    _dump_json(out_path, items)


def build_qa_compact(*, qa_path: Path, out_path: Path) -> None:
    qa = _load_json(qa_path)
    if not isinstance(qa, list):
        raise ValueError(f"Expected list QA JSON, got {type(qa).__name__}: {qa_path}")

    out: list[dict[str, Any]] = []
    for i, item in enumerate(qa):
        if not isinstance(item, dict):
            raise ValueError(f"qa[{i}] not a dict")

        qid = item.get("id") if item.get("id") is not None else item.get("_id")
        if qid is None:
            raise ValueError(f"qa[{i}] missing id/_id")

        question = item.get("question")
        if not isinstance(question, str):
            question = "" if question is None else str(question)

        gold_ans = item.get("gold_ans")
        answers = item.get("answers")
        answer = gold_ans
        if answer is None and isinstance(answers, list) and answers:
            answer = answers[0]
        if answer is None:
            answer = ""

        out.append(
            {
                "id": str(qid),
                "question": question,
                "answer": str(answer),
                "answers": answers if isinstance(answers, list) else None,
                "gold_ans": gold_ans,
                "dataset": item.get("dataset"),
                "language": item.get("language"),
                "length": item.get("length"),
                "all_classes": item.get("all_classes"),
                "distractor": item.get("distractor"),
            }
        )

    _dump_json(out_path, out)


def prepare_lveval_files(root: Path, *, rebuild: bool = False) -> dict[str, Path]:
    lveval_dir = root / "LVEVAL"
    corpus_path = lveval_dir / "lveval_corpus.json"
    qa_path = lveval_dir / "lveval.json"

    corpus_wrapper_path = lveval_dir / "lveval_corpus_for_pipeline.json"
    qa_compact_path = lveval_dir / "lveval_qa_compact.json"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}")
    if not qa_path.exists():
        raise FileNotFoundError(f"Missing QA: {qa_path}")

    if rebuild or not corpus_wrapper_path.exists():
        build_corpus_wrapper_for_pipeline(corpus_path=corpus_path, out_path=corpus_wrapper_path)

    if rebuild or not qa_compact_path.exists():
        build_qa_compact(qa_path=qa_path, out_path=qa_compact_path)

    return {
        "corpus": corpus_path,
        "qa": qa_path,
        "corpus_wrapper": corpus_wrapper_path,
        "qa_compact": qa_compact_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = prepare_lveval_files(root, rebuild=bool(args.rebuild))
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
