"""Prepare POPQA artifacts with exact-text duplicate passages canonicalized.

This does not rebuild BM25 or dense embeddings. It keeps row order and embedding
vectors intact, but rewrites per-row doc_id/title metadata so title variants with
identical passage text collapse to the same canonical doc_id during retrieval.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
POPQA = ROOT / "POPQA"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _doc_id(p: dict[str, Any], fallback: int) -> str:
    value = p.get("corpus_idx", p.get("idx", fallback))
    return str(value)


def _title(p: dict[str, Any]) -> str:
    return str(p.get("title") or "")


def _text(p: dict[str, Any]) -> str:
    return str(p.get("paragraph_text") or p.get("text") or "")


def _sort_doc_ids(ids: list[str]) -> list[str]:
    def key(x: str) -> tuple[int, int | str]:
        return (0, int(x)) if str(x).isdigit() else (1, str(x))

    return sorted(ids, key=key)


def build_canonical_map() -> dict[str, Any]:
    corpus = load_json(POPQA / "popqa_corpus_for_pipeline.json")[0]["paragraphs"]
    by_text: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_id: dict[str, dict[str, str]] = {}

    for i, p in enumerate(corpus):
        did = _doc_id(p, i)
        rec = {"doc_id": did, "title": _title(p), "text": _text(p)}
        by_text[rec["text"]].append(rec)
        by_id[did] = rec

    canonical_by_doc_id: dict[str, str] = {}
    canonical_title_by_doc_id: dict[str, str] = {}
    groups: list[dict[str, Any]] = []
    for text, rows in by_text.items():
        ids = _sort_doc_ids([r["doc_id"] for r in rows])
        canonical = ids[0]
        canonical_title = by_id[canonical]["title"]
        for did in ids:
            canonical_by_doc_id[did] = canonical
            canonical_title_by_doc_id[did] = canonical_title
        if len(ids) > 1:
            groups.append(
                {
                    "canonical_doc_id": canonical,
                    "canonical_title": canonical_title,
                    "doc_ids": ids,
                    "titles": [by_id[did]["title"] for did in ids],
                    "text": text,
                }
            )

    return {
        "dataset": "popqa",
        "rule": "exact same passage text -> lowest numeric doc_id is canonical",
        "total_docs": len(corpus),
        "duplicate_groups": len(groups),
        "duplicate_affected_docs": sum(len(g["doc_ids"]) for g in groups),
        "duplicate_extra_docs": sum(len(g["doc_ids"]) - 1 for g in groups),
        "canonical_by_doc_id": canonical_by_doc_id,
        "canonical_title_by_doc_id": canonical_title_by_doc_id,
        "groups": groups,
    }


def canonical_doc(mapping: dict[str, Any], doc_id: Any) -> str:
    s = str(doc_id)
    return str(mapping["canonical_by_doc_id"].get(s, s))


def canonical_title(mapping: dict[str, Any], doc_id: Any, current_title: Any) -> str:
    s = str(doc_id)
    return str(mapping["canonical_title_by_doc_id"].get(s, current_title or ""))


def write_embedding_texts(mapping: dict[str, Any]) -> None:
    data = load_json(POPQA / "facts.json")
    for row in data:
        did = row.get("doc_id")
        if did is None:
            continue
        row["doc_id"] = canonical_doc(mapping, did)
        ct = canonical_title(mapping, did, row.get("source_title") or row.get("title"))
        row["source_title"] = ct
        row["entity_title"] = ct
        row["title"] = ct
    write_json(POPQA / "facts_canonical.json", data)


def write_fact_embeddings(mapping: dict[str, Any]) -> None:
    src = POPQA / "fact_embeddings.npz"
    dst = POPQA / "fact_embeddings_canonical.npz"
    data = np.load(src, allow_pickle=True)
    arrays: dict[str, Any] = {k: data[k] for k in data.files}

    doc_ids = np.array([canonical_doc(mapping, x) for x in arrays["doc_ids"]], dtype=object)
    source_titles = np.array(
        [canonical_title(mapping, did, title) for did, title in zip(arrays["doc_ids"], arrays["source_titles"])],
        dtype=object,
    )
    entity_titles = np.array(
        [canonical_title(mapping, did, title) for did, title in zip(arrays["doc_ids"], arrays["entity_titles"])],
        dtype=object,
    )
    titles = np.array(
        [canonical_title(mapping, did, title) for did, title in zip(arrays["doc_ids"], arrays["titles"])],
        dtype=arrays["titles"].dtype,
    )

    arrays["doc_ids"] = doc_ids
    arrays["source_titles"] = source_titles
    arrays["entity_titles"] = entity_titles
    arrays["titles"] = titles
    np.savez_compressed(dst, **arrays)


def write_bm25_metadata(mapping: dict[str, Any]) -> None:
    src_dir = POPQA / "bm25_index"
    dst_dir = POPQA / "bm25_index_canonical"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.name == "metadata.json":
            continue
        dst = dst_dir / src.name
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)

    data = load_json(src_dir / "metadata.json")
    for row in data:
        did = row.get("doc_id")
        if did is None:
            continue
        row["doc_id"] = canonical_doc(mapping, did)
        ct = canonical_title(mapping, did, row.get("source_title") or row.get("title"))
        row["source_title"] = ct
        row["entity_title"] = ct
        row["title"] = ct
    write_json(dst_dir / "metadata.json", data)


def write_metadata_json_and_db(mapping: dict[str, Any]) -> None:
    meta = load_json(POPQA / "metadata.json")
    for item in meta:
        filtered = []
        seen = set()
        for row in item.get("context_metadata", []) or []:
            did = str(row.get("doc_id"))
            canon = canonical_doc(mapping, did)
            if did != canon:
                continue
            if canon in seen:
                continue
            seen.add(canon)
            filtered.append(row)
        item["context_metadata"] = filtered
    write_json(POPQA / "metadata_canonical.json", meta)

    src = POPQA / "metadata.db"
    dst = POPQA / "metadata_canonical.db"
    if dst.exists():
        dst.unlink()
    src_conn = sqlite3.connect(src)
    src_conn.row_factory = sqlite3.Row
    dst_conn = sqlite3.connect(dst)
    dst_conn.execute(
        "CREATE TABLE metadata (doc_id TEXT PRIMARY KEY, source_title TEXT, entity_title TEXT, metadata_json TEXT)"
    )
    seen = set()
    for row in src_conn.execute("SELECT doc_id, source_title, entity_title, metadata_json FROM metadata"):
        did = str(row["doc_id"])
        canon = canonical_doc(mapping, did)
        if did != canon or canon in seen:
            continue
        seen.add(canon)
        dst_conn.execute(
            "INSERT INTO metadata (doc_id, source_title, entity_title, metadata_json) VALUES (?, ?, ?, ?)",
            (canon, row["source_title"], row["entity_title"], row["metadata_json"]),
        )
    dst_conn.commit()
    dst_conn.close()
    src_conn.close()


def main() -> None:
    mapping = build_canonical_map()
    write_json(POPQA / "canonical_doc_id_map.json", mapping)
    write_embedding_texts(mapping)
    write_fact_embeddings(mapping)
    write_bm25_metadata(mapping)
    write_metadata_json_and_db(mapping)

    print("Wrote POPQA canonical artifacts")
    print(f"duplicate_groups={mapping['duplicate_groups']}")
    print(f"duplicate_affected_docs={mapping['duplicate_affected_docs']}")
    print(f"duplicate_extra_docs={mapping['duplicate_extra_docs']}")


if __name__ == "__main__":
    main()
