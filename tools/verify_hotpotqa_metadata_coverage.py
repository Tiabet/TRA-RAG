from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


def _load_corpus_doc_ids(corpus_path: Path) -> tuple[set[str], dict[str, str]]:
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    doc_ids: set[str] = set()
    titles_by_id: dict[str, str] = {}
    for item in data:
        idx = item.get("idx")
        if idx is None:
            continue
        doc_id = str(idx)
        doc_ids.add(doc_id)
        title = item.get("title")
        if isinstance(title, str):
            titles_by_id[doc_id] = title
    return doc_ids, titles_by_id


def _iter_db_doc_ids(conn: sqlite3.Connection) -> tuple[str, str, list[str]]:
    """Return (table_name, column_name, doc_ids). Picks the best candidate table/column."""

    def get_tables() -> list[str]:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    def get_columns(table: str) -> list[str]:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [r[1] for r in rows]

    tables = get_tables()
    if not tables:
        raise RuntimeError("No tables found in metadata DB")

    # Prefer a table that looks like it stores per-document metadata.
    preferred_table_names = [
        "metadata",
        "doc_metadata",
        "documents",
        "docs",
        "passages",
        "passage_metadata",
    ]

    table_candidates = preferred_table_names + tables

    chosen_table = None
    chosen_col = None

    col_preferences = ["doc_id", "idx", "corpus_idx", "document_id", "id"]

    for t in table_candidates:
        if t not in tables:
            continue
        cols = get_columns(t)
        for c in col_preferences:
            if c in cols:
                chosen_table = t
                chosen_col = c
                break
        if chosen_table:
            break

    if not chosen_table or not chosen_col:
        # Fall back: find any table with a plausible id column
        for t in tables:
            cols = get_columns(t)
            for c in cols:
                if c.lower() in col_preferences:
                    chosen_table, chosen_col = t, c
                    break
            if chosen_table:
                break

    if not chosen_table or not chosen_col:
        raise RuntimeError(f"Could not find a doc id column in tables: {tables}")

    rows = conn.execute(f"SELECT {chosen_col} FROM {chosen_table}").fetchall()
    doc_ids = [str(r[0]) for r in rows if r and r[0] is not None]
    return chosen_table, chosen_col, doc_ids


def _maybe_title_column(conn: sqlite3.Connection, table: str) -> str | None:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    for c in ["title", "doc_title", "page_title"]:
        if c in cols:
            return c
    return None


def _sample(items: Iterable[str], n: int = 20) -> list[str]:
    out: list[str] = []
    for x in items:
        out.append(x)
        if len(out) >= n:
            break
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    corpus_path = root / "HotpotQA" / "hotpotqa_corpus.json"
    db_path = root / "HotpotQA" / "metadata_v5.db"

    if not corpus_path.exists():
        raise FileNotFoundError(str(corpus_path))
    if not db_path.exists():
        raise FileNotFoundError(str(db_path))

    corpus_ids, corpus_titles = _load_corpus_doc_ids(corpus_path)

    conn = sqlite3.connect(str(db_path))
    try:
        table, col, db_ids_list = _iter_db_doc_ids(conn)
        db_ids = set(db_ids_list)

        missing = sorted(corpus_ids - db_ids, key=lambda x: int(x) if x.isdigit() else x)
        extra = sorted(db_ids - corpus_ids, key=lambda x: int(x) if x.isdigit() else x)

        # Duplicates in DB
        dup_rows = conn.execute(
            f"SELECT {col}, COUNT(*) as c FROM {table} GROUP BY {col} HAVING c > 1 ORDER BY c DESC LIMIT 20"
        ).fetchall()

        title_col = _maybe_title_column(conn, table)
        title_mismatch_samples: list[str] = []
        if title_col is not None:
            # Sample a few rows where title differs from corpus title (only when doc_id is numeric)
            rows = conn.execute(
                f"SELECT {col}, {title_col} FROM {table} WHERE {col} IS NOT NULL LIMIT 20000"
            ).fetchall()
            for doc_id_raw, db_title in rows:
                doc_id = str(doc_id_raw)
                if doc_id not in corpus_titles:
                    continue
                if isinstance(db_title, str) and db_title != corpus_titles[doc_id]:
                    title_mismatch_samples.append(
                        f"{doc_id}: corpus='{corpus_titles[doc_id]}' db='{db_title}'"
                    )
                    if len(title_mismatch_samples) >= 20:
                        break

        print("=== HotpotQA metadata coverage check ===")
        print(f"corpus: {corpus_path}")
        print(f"metadata db: {db_path}")
        print(f"db table/col: {table}.{col}")
        print("")
        print(f"corpus_doc_ids: {len(corpus_ids)}")
        print(f"db_doc_ids: {len(db_ids)} (rows={len(db_ids_list)})")
        print(f"missing_in_db: {len(missing)}")
        print(f"extra_in_db: {len(extra)}")
        print(f"duplicate_doc_ids_in_db(top20): {len(dup_rows)}")
        if title_col is not None:
            print(f"title_col: {title_col}")
            print(f"title_mismatch_samples: {len(title_mismatch_samples)}")
        else:
            print("title_col: (none found)")
        print("")

        if missing:
            print("-- missing sample --")
            for doc_id in _sample(missing, 30):
                title = corpus_titles.get(doc_id, "")
                if title:
                    print(f"{doc_id}\t{title}")
                else:
                    print(doc_id)
            print("")

        if extra:
            print("-- extra sample --")
            for doc_id in _sample(extra, 30):
                print(doc_id)
            print("")

        if dup_rows:
            print("-- duplicates (doc_id, count) --")
            for doc_id, c in dup_rows:
                print(doc_id, c)
            print("")

        if title_mismatch_samples:
            print("-- title mismatch sample --")
            for s in title_mismatch_samples:
                print(s)
            print("")

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
