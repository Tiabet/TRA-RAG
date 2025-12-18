#!/usr/bin/env python3
"""Verify doc_id alignment across artifacts.

Checks that doc_id values used by:
- original dataset JSON (e.g., HotpotQA/hotpotqa_sample_200.json)
- embedding_texts.json (metadata-path corpus)
- metadata DB (metadata_v3.db)
- result files (Results/*.json)

are consistent and refer to the same (id, ctx_idx) passages.

This script is intentionally lightweight (stdlib only).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class DocIdStats:
    total: int
    missing_in_other: int
    extra_in_other: int


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_docid_to_title_from_dataset(dataset_json: Path) -> Dict[str, str]:
    """Build doc_id -> title from the dataset's context list.

    Expects each item to have '_id' and 'context' as list of [title, sentences].
    """
    data = _load_json(dataset_json)
    mapping: Dict[str, str] = {}
    for item in data:
        sample_id = item.get("_id")
        if not sample_id:
            continue
        context = item.get("context", []) or []
        for ctx_idx, ctx in enumerate(context):
            if not isinstance(ctx, list) or len(ctx) < 1:
                continue
            title = ctx[0]
            doc_id = f"{sample_id}::ctx{ctx_idx}"
            # If duplicated doc_id ever happens (shouldn't), keep first.
            mapping.setdefault(doc_id, str(title))
    return mapping


def iter_embedding_texts_rows(
    embedding_texts_json: Path,
    max_rows: Optional[int] = None,
) -> Iterable[Tuple[str, Optional[str], Optional[str], Optional[str]]]:
    """Iterate (doc_id, title, source_title, entity_title) from embedding_texts.json."""
    data = _load_json(embedding_texts_json)
    count = 0
    for row in data:
        if max_rows is not None and count >= max_rows:
            break
        count += 1
        if not isinstance(row, dict):
            continue
        doc_id = row.get("doc_id")
        if doc_id is None:
            continue
        title = row.get("title")
        source_title = row.get("source_title")
        entity_title = row.get("entity_title")
        yield (
            str(doc_id),
            (str(title) if title is not None else None),
            (str(source_title) if source_title is not None else None),
            (str(entity_title) if entity_title is not None else None),
        )


def load_db_docid_to_title(db_path: Path) -> Dict[str, Optional[str]]:
    """Load doc_id -> title from SQLite metadata DB (best-effort).

    The DB schema varies slightly across versions; we try common columns.
    """
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        # Detect columns
        cur.execute("PRAGMA table_info(metadata)")
        cols = [r["name"] for r in cur.fetchall()]
        if "doc_id" not in cols:
            return {}

        title_col = "title" if "title" in cols else None
        if title_col:
            cur.execute("SELECT doc_id, title FROM metadata")
            rows = cur.fetchall()
            return {str(r["doc_id"]): (str(r["title"]) if r["title"] is not None else None) for r in rows}

        cur.execute("SELECT doc_id FROM metadata")
        rows = cur.fetchall()
        return {str(r["doc_id"]): None for r in rows}
    finally:
        conn.close()


def collect_result_doc_ids(result_json: Path) -> Set[str]:
    """Collect doc_ids from a result file (top-level + decomposition)."""
    data = _load_json(result_json)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        return set()

    doc_ids: Set[str] = set()

    def add(v):
        if not v:
            return
        doc_ids.add(str(v))

    for item in data:
        if not isinstance(item, dict):
            continue

        for p in item.get("retrieved_passages", []) or []:
            if isinstance(p, dict) and p.get("doc_id") is not None:
                add(p.get("doc_id"))

        for p in item.get("retrieved_paths", []) or []:
            if isinstance(p, dict) and p.get("doc_id") is not None:
                add(p.get("doc_id"))

        decomp = item.get("decomposition")
        subqs = []
        if isinstance(decomp, dict):
            subqs = decomp.get("subquestions", []) or []
        elif isinstance(decomp, list):
            subqs = decomp

        for sq in subqs:
            if not isinstance(sq, dict):
                continue
            for p in sq.get("retrieved_passages", []) or []:
                if isinstance(p, dict) and p.get("doc_id") is not None:
                    add(p.get("doc_id"))
            for p in sq.get("retrieved_paths", []) or []:
                if isinstance(p, dict) and p.get("doc_id") is not None:
                    add(p.get("doc_id"))

    return doc_ids


def analyze_duplicate_titles_in_context(dataset_json: Path) -> Dict[str, int]:
    """Analyze per-item duplicate titles in the context.

    This is the core ambiguity risk for title→doc_id mapping:
    if a title occurs multiple times in a single item's context, mapping a title to a
    unique doc_id becomes ambiguous.
    """
    data = _load_json(dataset_json)
    items = 0
    items_with_dupes = 0
    total_dupe_titles = 0
    max_dupe_count_for_a_title = 0

    for item in data:
        items += 1
        counts: Dict[str, int] = {}
        for ctx in item.get("context", []) or []:
            if not isinstance(ctx, list) or not ctx:
                continue
            title = str(ctx[0])
            counts[title] = counts.get(title, 0) + 1
        dupes = {t: c for t, c in counts.items() if c > 1}
        if dupes:
            items_with_dupes += 1
            total_dupe_titles += len(dupes)
            max_dupe_count_for_a_title = max(max_dupe_count_for_a_title, max(dupes.values()))

    return {
        "items": items,
        "items_with_duplicate_titles_in_context": items_with_dupes,
        "total_duplicate_titles_across_items": total_dupe_titles,
        "max_duplicate_count_for_single_title_within_item": max_dupe_count_for_a_title,
    }


def analyze_result_path_to_passage_docid_consistency(result_json: Path) -> Dict[str, int]:
    """Check whether retrieved_passages doc_ids are backed by retrieved_paths doc_ids.

    For SQ-level passage selection (your current policy), every passage doc_id should
    usually appear among that SQ's retrieved_paths doc_ids.
    """
    data = _load_json(result_json)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        return {
            "items": 0,
            "subquestions": 0,
            "sq_passage_doc_ids": 0,
            "sq_passage_doc_ids_missing_in_sq_paths": 0,
            "top_level_passage_doc_ids": 0,
            "top_level_passage_doc_ids_missing_in_top_level_paths": 0,
        }

    items = 0
    subquestions = 0
    sq_passage_doc_ids = 0
    sq_missing = 0
    top_passage_doc_ids = 0
    top_missing = 0

    def _doc_id_set_from_paths(paths) -> Set[str]:
        s: Set[str] = set()
        for p in paths or []:
            if isinstance(p, dict) and p.get("doc_id") is not None:
                s.add(str(p.get("doc_id")))
        return s

    def _doc_id_list_from_passages(passages) -> List[str]:
        out: List[str] = []
        for p in passages or []:
            if isinstance(p, dict) and p.get("doc_id") is not None:
                out.append(str(p.get("doc_id")))
        return out

    for item in data:
        if not isinstance(item, dict):
            continue
        items += 1

        # Top-level
        top_paths = _doc_id_set_from_paths(item.get("retrieved_paths"))
        for d in _doc_id_list_from_passages(item.get("retrieved_passages")):
            top_passage_doc_ids += 1
            if d not in top_paths and top_paths:
                top_missing += 1

        # SQ-level
        decomp = item.get("decomposition")
        subqs = []
        if isinstance(decomp, dict):
            subqs = decomp.get("subquestions", []) or []
        elif isinstance(decomp, list):
            subqs = decomp
        for sq in subqs:
            if not isinstance(sq, dict):
                continue
            subquestions += 1
            sq_paths = _doc_id_set_from_paths(sq.get("retrieved_paths"))
            for d in _doc_id_list_from_passages(sq.get("retrieved_passages")):
                sq_passage_doc_ids += 1
                # Only count as missing if we actually have paths recorded
                if d not in sq_paths and sq_paths:
                    sq_missing += 1

    return {
        "items": items,
        "subquestions": subquestions,
        "sq_passage_doc_ids": sq_passage_doc_ids,
        "sq_passage_doc_ids_missing_in_sq_paths": sq_missing,
        "top_level_passage_doc_ids": top_passage_doc_ids,
        "top_level_passage_doc_ids_missing_in_top_level_paths": top_missing,
    }


def compare_sets(a: Set[str], b: Set[str]) -> Tuple[int, int, int]:
    """Return (|a|, missing_in_b, extra_in_b)."""
    missing = len(a - b)
    extra = len(b - a)
    return len(a), missing, extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_json", required=True, help="e.g., HotpotQA/hotpotqa_sample_200.json")
    ap.add_argument("--embedding_texts", required=True, help="e.g., HotpotQA/embedding_texts.json")
    ap.add_argument("--db_path", default=None, help="e.g., HotpotQA/metadata_v3.db")
    ap.add_argument("--result_json", default=None, help="optional Results/*.json to validate retrieved doc_ids")
    ap.add_argument("--max_embedding_rows", type=int, default=None, help="optional cap for embedding_texts.json rows")

    args = ap.parse_args()

    dataset_json = Path(args.dataset_json)
    embedding_texts = Path(args.embedding_texts)
    db_path = Path(args.db_path) if args.db_path else None
    result_json = Path(args.result_json) if args.result_json else None

    if not dataset_json.exists():
        raise FileNotFoundError(dataset_json)
    if not embedding_texts.exists():
        raise FileNotFoundError(embedding_texts)

    docid_to_title = build_docid_to_title_from_dataset(dataset_json)
    dataset_doc_ids = set(docid_to_title.keys())

    emb_doc_ids: Set[str] = set()
    emb_docid_to_title: Dict[str, Optional[str]] = {}
    emb_docid_to_source_title: Dict[str, Optional[str]] = {}
    emb_docid_to_entity_title: Dict[str, Optional[str]] = {}

    for doc_id, title, source_title, entity_title in iter_embedding_texts_rows(
        embedding_texts, max_rows=args.max_embedding_rows
    ):
        emb_doc_ids.add(doc_id)
        if doc_id not in emb_docid_to_title and title is not None:
            emb_docid_to_title[doc_id] = title
        if doc_id not in emb_docid_to_source_title and source_title is not None:
            emb_docid_to_source_title[doc_id] = source_title
        if doc_id not in emb_docid_to_entity_title and entity_title is not None:
            emb_docid_to_entity_title[doc_id] = entity_title

    print("\n[1] Dataset vs embedding_texts.json doc_id coverage")
    total, missing_in_emb, extra_in_emb = compare_sets(dataset_doc_ids, emb_doc_ids)
    print(f"Dataset doc_ids: {total}")
    print(f"Missing in embedding_texts: {missing_in_emb}")
    print(f"Extra in embedding_texts (not in dataset): {extra_in_emb}")

    # Title consistency on intersection (dataset context title vs embedding_texts fields)
    inter = dataset_doc_ids & emb_doc_ids
    checked = 0
    match_title = 0
    match_source_title = 0
    match_entity_title = 0
    for d in inter:
        t_dataset = docid_to_title.get(d)
        checked += 1

        if t_dataset is None:
            continue
        if emb_docid_to_title.get(d) is not None and str(t_dataset) == str(emb_docid_to_title.get(d)):
            match_title += 1
        if emb_docid_to_source_title.get(d) is not None and str(t_dataset) == str(emb_docid_to_source_title.get(d)):
            match_source_title += 1
        if emb_docid_to_entity_title.get(d) is not None and str(t_dataset) == str(emb_docid_to_entity_title.get(d)):
            match_entity_title += 1

    if checked:
        print(
            "Title match rates on doc_id intersection:\n"
            f"- dataset title == embedding_texts.title: {match_title}/{checked}\n"
            f"- dataset title == embedding_texts.source_title: {match_source_title}/{checked}\n"
            f"- dataset title == embedding_texts.entity_title: {match_entity_title}/{checked}"
        )

    print("\n[1.1] Duplicate-title ambiguity in dataset context")
    dupe_stats = analyze_duplicate_titles_in_context(dataset_json)
    print(f"Items: {dupe_stats['items']}")
    print(f"Items with duplicate titles in context: {dupe_stats['items_with_duplicate_titles_in_context']}")
    print(f"Total duplicate titles across items: {dupe_stats['total_duplicate_titles_across_items']}")
    print(f"Max duplicate count for a single title within an item: {dupe_stats['max_duplicate_count_for_single_title_within_item']}")

    if db_path is not None:
        print("\n[2] DB metadata_v3.db doc_id coverage")
        db_docid_to_title = load_db_docid_to_title(db_path)
        db_doc_ids = set(db_docid_to_title.keys())
        total, missing_in_db, extra_in_db = compare_sets(dataset_doc_ids, db_doc_ids)
        print(f"DB doc_ids: {len(db_doc_ids)}")
        print(f"Missing in DB: {missing_in_db}")
        print(f"Extra in DB (not in dataset): {extra_in_db}")

        # DB title consistency if available
        db_checked = 0
        db_mismatch = 0
        for d in (dataset_doc_ids & db_doc_ids):
            t_db = db_docid_to_title.get(d)
            if t_db is None:
                continue
            db_checked += 1
            if str(docid_to_title.get(d)) != str(t_db):
                db_mismatch += 1
        if db_checked:
            print(f"Title consistency (dataset title == DB title) on {db_checked} doc_ids: {(db_checked - db_mismatch)}/{db_checked} OK")
            if db_mismatch:
                print(f"Title mismatches: {db_mismatch}")

    if result_json is not None:
        print("\n[3] Results file doc_id validity")
        result_doc_ids = collect_result_doc_ids(result_json)
        print(f"Unique retrieved doc_ids in result: {len(result_doc_ids)}")
        missing_in_dataset = len(result_doc_ids - dataset_doc_ids)
        missing_in_emb = len(result_doc_ids - emb_doc_ids)
        print(f"Retrieved doc_ids missing in dataset_json: {missing_in_dataset}")
        print(f"Retrieved doc_ids missing in embedding_texts: {missing_in_emb}")

        print("\n[3.1] Results path→passage doc_id consistency")
        cons = analyze_result_path_to_passage_docid_consistency(result_json)
        print(f"Items: {cons['items']}")
        print(f"Subquestions: {cons['subquestions']}")
        if cons['sq_passage_doc_ids']:
            print(
                f"SQ passage doc_ids backed by SQ paths: {cons['sq_passage_doc_ids'] - cons['sq_passage_doc_ids_missing_in_sq_paths']}/{cons['sq_passage_doc_ids']}"
            )
        if cons['top_level_passage_doc_ids']:
            print(
                f"Top-level passage doc_ids backed by top-level paths: {cons['top_level_passage_doc_ids'] - cons['top_level_passage_doc_ids_missing_in_top_level_paths']}/{cons['top_level_passage_doc_ids']}"
            )

        # Print a few examples if bad
        if missing_in_dataset:
            examples = list(result_doc_ids - dataset_doc_ids)[:10]
            print("Examples missing in dataset:")
            for ex in examples:
                print(f"  - {ex}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
