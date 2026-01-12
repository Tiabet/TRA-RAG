#!/usr/bin/env python3
"""Merge 2Wiki QA (question/answer) into paragraphs-only JSON by id.

Inputs:
- 2WikiMultihopQA/2wiki_qa.json: has _id, question, answer, type, etc.
- 2WikiMultihopQA/2wikimultihopqa.json: has id, paragraphs

Output:
- Updates 2WikiMultihopQA/2wikimultihopqa.json in-place (makes a .bak first)

This keeps the existing paragraphs structure used by the pipeline, and adds:
- _id (same as id)
- question
- answer
- answer_aliases (empty list if not present)
- type (if present)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_str_or_none(v: Any):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="2WikiMultihopQA/2wiki_qa.json")
    ap.add_argument("--paragraphs", default="2WikiMultihopQA/2wikimultihopqa.json")
    ap.add_argument("--inplace", action="store_true", help="Write back to --paragraphs (creates .bak)")
    ap.add_argument("--out", default="", help="Optional output path (if not inplace)")
    args = ap.parse_args()

    qa_path = Path(args.qa)
    para_path = Path(args.paragraphs)
    if not qa_path.exists():
        raise FileNotFoundError(qa_path)
    if not para_path.exists():
        raise FileNotFoundError(para_path)

    qa_data = _load_json(qa_path)
    para_data = _load_json(para_path)

    if not isinstance(qa_data, list):
        raise ValueError(f"Expected list: {qa_path}")
    if not isinstance(para_data, list):
        raise ValueError(f"Expected list: {para_path}")

    qa_by_id: Dict[str, Dict[str, Any]] = {}
    dup = 0
    for item in qa_data:
        if not isinstance(item, dict):
            continue
        qid = item.get("_id") or item.get("id")
        if qid is None:
            continue
        qid = str(qid)
        if qid in qa_by_id:
            dup += 1
            continue
        qa_by_id[qid] = item

    updated = 0
    missing = 0

    for item in para_data:
        if not isinstance(item, dict):
            continue
        pid = item.get("id") or item.get("_id")
        if pid is None:
            missing += 1
            continue
        pid = str(pid)
        qa = qa_by_id.get(pid)
        if not qa:
            missing += 1
            continue

        # Add fields expected by evaluation / runners
        item["_id"] = pid
        item["question"] = _to_str_or_none(qa.get("question")) or ""
        item["answer"] = _to_str_or_none(qa.get("answer"))
        item["gold_answer"] = _to_str_or_none(qa.get("answer"))
        item["answer_aliases"] = []
        if "type" in qa:
            item["type"] = qa.get("type")

        updated += 1

    if args.inplace:
        backup = para_path.with_suffix(para_path.suffix + ".bak")
        shutil.copyfile(para_path, backup)
        _write_json(para_path, para_data)
        print(f"[OK] Wrote updated dataset in-place: {para_path} (backup: {backup})")
    else:
        out = Path(args.out) if args.out else para_path.with_name(para_path.stem + ".merged.json")
        _write_json(out, para_data)
        print(f"[OK] Wrote merged dataset: {out}")

    print(f"QA items: {len(qa_data)}")
    print(f"Paragraph items: {len(para_data)}")
    print(f"Updated: {updated}")
    print(f"Missing QA match: {missing}")
    if dup:
        print(f"[WARN] Duplicate QA ids skipped: {dup}")

    # Basic sanity
    if updated == 0:
        return 2
    if missing:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
