#!/usr/bin/env python3
"""Build combined v5 artifacts for large-scale robustness.

Goal
- Create ONE merged set of v5 artifacts (DB + BM25 + dense path embeddings)
  by *reusing existing per-dataset artifacts* and applying corpus_idx offsets
  to avoid doc_id collisions.
- Create a single JSON file that contains:
  - 2wiki QA items (for evaluation)
  - additional corpus-only paragraphs from other datasets (for passage lookup)

Why offsets are required
- All four datasets in this repo use numeric doc_ids starting at 0.
- If we concatenate artifacts without rewriting doc_id, doc_id collisions corrupt
  DB lookups and passage rendering.

Outputs (default under RobustnessCombined/)
- RobustnessCombined/offsets.json
- RobustnessCombined/metadata_v5.db
- RobustnessCombined/bm25_index_v5/
- RobustnessCombined/path_embeddings_v5.npz
- RobustnessCombined/2wiki_qa_plus_4corpus.json

This script intentionally does NOT change any pipeline logic.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

import bm25s
import Stemmer


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    db_path: Path
    bm25_dir: Path
    path_embeddings_npz: Path

    # For passage corpus injection into the combined QA JSON
    # We will transform these corpora into MuSiQue-style paragraphs (title, paragraph_text, corpus_idx).
    corpus_kind: str  # '2wiki_qa' | 'hotpot_corpus' | 'musique_qa' | 'lveval_corpus'
    corpus_path: Path


STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with','by','from','as','is','was','are','were','been',
    'be','have','has','had','do','does','did','will','would','could','should','may','might','must','shall','can','need',
    'it','its','this','that','these','those','i','you','he','she','we','they','what','which','who','whom','whose',
    'where','when','why','how','all','each','every','both','few','more','most','other','some','such','no','nor','not',
    'only','own','same','so','than','too','very','just','also'
}


def _preprocess_for_bm25(text: str, *, stemmer: Stemmer.Stemmer) -> list[str]:
    import re

    if not text:
        return []

    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    toks = [x for x in t.split() if x and (x not in STOPWORDS) and (len(x) > 1)]
    try:
        toks = stemmer.stemWords(toks)
    except Exception:
        pass
    return toks


def _build_bm25_text(title: str, key_path: str, value: str) -> str:
    kp = (key_path or '').replace('.', ' ')
    return f"{title or ''} {kp} {value or ''}".strip()


def _max_doc_id_from_db(db_path: Path) -> int:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    m = cur.execute("SELECT MAX(CAST(doc_id AS INTEGER)) FROM metadata").fetchone()[0]
    con.close()
    if m is None:
        raise RuntimeError(f"DB has no rows: {db_path}")
    return int(m)


def _iter_db_rows(db_path: Path) -> Iterator[tuple[str, str, str, str]]:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    for row in cur.execute("SELECT doc_id, source_title, entity_title, metadata_json FROM metadata"):
        yield (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
    con.close()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json_list_stream(path: Path, items: Iterable[Any]) -> None:
    """Write a JSON array incrementally to avoid holding everything in memory."""
    with path.open('w', encoding='utf-8') as f:
        f.write('[\n')
        first = True
        for it in items:
            if not first:
                f.write(',\n')
            first = False
            json.dump(it, f, ensure_ascii=False)
        f.write('\n]\n')


def _chunks(seq: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _build_offsets(datasets: dict[str, DatasetPaths]) -> dict[str, int]:
    # Keep 2wiki at offset 0 so gold corpus_idx remains unchanged.
    order = ['2wiki', 'hotpot', 'musique', 'lveval']
    max_ids = {k: _max_doc_id_from_db(datasets[k].db_path) for k in order}
    offsets: dict[str, int] = {'2wiki': 0}

    cur = max_ids['2wiki'] + 1
    for k in order[1:]:
        offsets[k] = int(cur)
        cur += int(max_ids[k] + 1)

    return offsets


def _merge_metadata_db(out_db: Path, datasets: dict[str, DatasetPaths], offsets: dict[str, int], rebuild: bool) -> None:
    if out_db.exists() and not rebuild:
        return

    if out_db.exists():
        out_db.unlink()

    con_out = sqlite3.connect(str(out_db))
    cur_out = con_out.cursor()
    cur_out.execute(
        "CREATE TABLE metadata (doc_id TEXT, source_title TEXT, entity_title TEXT, metadata_json TEXT)"
    )
    cur_out.execute("CREATE INDEX idx_metadata_doc_id ON metadata(doc_id)")

    def insert_many(rows: list[tuple[str, str, str, str]]):
        cur_out.executemany(
            "INSERT INTO metadata (doc_id, source_title, entity_title, metadata_json) VALUES (?, ?, ?, ?)",
            rows,
        )

    for key, ds in datasets.items():
        off = int(offsets[key])
        batch: list[tuple[str, str, str, str]] = []
        for doc_id, source_title, entity_title, metadata_json in _iter_db_rows(ds.db_path):
            try:
                new_id = str(int(doc_id) + off)
            except Exception:
                raise RuntimeError(f"Non-numeric doc_id in {ds.db_path}: {doc_id}")
            batch.append((new_id, source_title, entity_title, metadata_json))
            if len(batch) >= 5000:
                insert_many(batch)
                con_out.commit()
                batch = []
        if batch:
            insert_many(batch)
            con_out.commit()

    con_out.close()


def _load_npz_arrays(path: Path) -> dict[str, Any]:
    return dict(np.load(str(path), allow_pickle=True))


def _merge_path_embeddings(out_npz: Path, datasets: dict[str, DatasetPaths], offsets: dict[str, int], rebuild: bool) -> dict[str, Any]:
    """Merge per-dataset path embeddings into one NPZ.

    Returns a dict with keys used later for BM25 build:
      - titles, key_paths, values, doc_ids, source_titles, entity_titles
    """

    if out_npz.exists() and not rebuild:
        # Load just metadata arrays for downstream steps
        data = np.load(str(out_npz), allow_pickle=True)
        return {
            'titles': list(data['titles']),
            'key_paths': list(data['key_paths']),
            'values': list(data['values']),
            'doc_ids': list(data['doc_ids']) if 'doc_ids' in data.files else [],
            'source_titles': list(data['source_titles']) if 'source_titles' in data.files else [None] * len(data['titles']),
            'entity_titles': list(data['entity_titles']) if 'entity_titles' in data.files else [None] * len(data['titles']),
        }

    # First pass: shapes
    parts = []
    total = 0
    dim = None
    for key, ds in datasets.items():
        npz = np.load(str(ds.path_embeddings_npz), allow_pickle=True)
        emb = npz['embeddings']
        if dim is None:
            dim = int(emb.shape[1])
        elif int(emb.shape[1]) != int(dim):
            raise RuntimeError(f"Embedding dim mismatch: {ds.path_embeddings_npz} has {emb.shape[1]} vs {dim}")
        n = int(emb.shape[0])
        parts.append((key, ds, n))
        total += n

    if dim is None or total <= 0:
        raise RuntimeError("No embeddings found to merge")

    _ensure_dir(out_npz.parent)

    # Disk-backed memmap for embeddings to keep peak RAM down.
    tmp_npy = out_npz.with_suffix('.embeddings.tmp.npy')
    if tmp_npy.exists():
        tmp_npy.unlink()
    emb_out = np.lib.format.open_memmap(str(tmp_npy), mode='w+', dtype=np.float32, shape=(total, dim))

    titles: list[str] = []
    key_paths: list[str] = []
    values: list[str] = []
    doc_ids: list[str] = []
    source_titles: list[str | None] = []
    entity_titles: list[str | None] = []

    cursor = 0
    for key, ds, n in parts:
        off = int(offsets[key])
        data = np.load(str(ds.path_embeddings_npz), allow_pickle=True)
        emb = data['embeddings'].astype(np.float32, copy=False)
        emb_out[cursor:cursor + n, :] = emb

        t = [str(x) for x in list(data['titles'])]
        kp = [str(x) for x in list(data['key_paths'])]
        v = [str(x) for x in list(data['values'])]
        d_raw = list(data['doc_ids']) if 'doc_ids' in data.files else [None] * n
        st_raw = list(data['source_titles']) if 'source_titles' in data.files else [None] * n
        et_raw = list(data['entity_titles']) if 'entity_titles' in data.files else [None] * n

        if len(t) != n or len(kp) != n or len(v) != n:
            raise RuntimeError(f"Metadata length mismatch in {ds.path_embeddings_npz}")

        titles.extend(t)
        key_paths.extend(kp)
        values.extend(v)

        for x in d_raw:
            if x is None:
                doc_ids.append('')
                continue
            try:
                doc_ids.append(str(int(str(x)) + off))
            except Exception:
                raise RuntimeError(f"Non-numeric doc_id in {ds.path_embeddings_npz}: {x}")

        source_titles.extend([None if (x is None) else str(x) for x in st_raw])
        entity_titles.extend([None if (x is None) else str(x) for x in et_raw])

        cursor += n

    if cursor != total:
        raise RuntimeError(f"Internal error: wrote {cursor} rows, expected {total}")

    # Flush memmap to disk before packaging.
    try:
        emb_out.flush()
    except Exception:
        pass

    # Write NPZ. We keep object arrays for compatibility with existing loader.
    # NOTE: np.savez_compressed will read from the memmap; this can take time.
    np.savez_compressed(
        str(out_npz),
        embeddings=emb_out,
        titles=np.asarray(titles, dtype=object),
        key_paths=np.asarray(key_paths, dtype=object),
        values=np.asarray(values, dtype=object),
        doc_ids=np.asarray(doc_ids, dtype=object),
        source_titles=np.asarray(source_titles, dtype=object),
        entity_titles=np.asarray(entity_titles, dtype=object),
    )

    # Cleanup temp memmap file (important on Windows to free disk space)
    try:
        # Ensure memmap file handle is released
        del emb_out
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        tmp_npy.unlink()
    except Exception:
        # Best-effort; caller can delete manually if the OS still holds a handle.
        pass

    return {
        'titles': titles,
        'key_paths': key_paths,
        'values': values,
        'doc_ids': doc_ids,
        'source_titles': source_titles,
        'entity_titles': entity_titles,
    }


def _build_bm25_index(
    out_dir: Path,
    meta_arrays: dict[str, Any],
    rebuild: bool,
) -> None:
    meta_path = out_dir / 'metadata.json'
    if meta_path.exists() and not rebuild:
        return

    if out_dir.exists() and rebuild:
        # bm25s writes multiple files; clear to avoid mixing.
        for child in out_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                # best-effort
                import shutil

                shutil.rmtree(child, ignore_errors=True)

    _ensure_dir(out_dir)

    titles = meta_arrays['titles']
    key_paths = meta_arrays['key_paths']
    values = meta_arrays['values']
    doc_ids = meta_arrays['doc_ids']
    source_titles = meta_arrays.get('source_titles')
    entity_titles = meta_arrays.get('entity_titles')

    n = len(titles)
    stemmer = Stemmer.Stemmer('english')

    # Build tokens
    corpus_tokens: list[list[str]] = []
    corpus_tokens.reserve(n) if hasattr(corpus_tokens, 'reserve') else None  # no-op on CPython

    for i in range(n):
        text = _build_bm25_text(str(titles[i]), str(key_paths[i]), str(values[i]))
        corpus_tokens.append(_preprocess_for_bm25(text, stemmer=stemmer))

    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)
    bm25.save(str(out_dir))

    def iter_metadata_dicts() -> Iterator[dict[str, Any]]:
        for i in range(n):
            yield {
                'title': str(titles[i]),
                'key_path': str(key_paths[i]),
                'value': str(values[i]),
                'doc_id': str(doc_ids[i]),
                'source_title': None if source_titles is None else (None if source_titles[i] is None else str(source_titles[i])),
                'entity_title': None if entity_titles is None else (None if entity_titles[i] is None else str(entity_titles[i])),
            }

    _write_json_list_stream(meta_path, iter_metadata_dicts())


def _iter_hotpot_corpus_paragraphs(corpus_path: Path, offset: int) -> Iterator[dict[str, Any]]:
    data = json.loads(corpus_path.read_text(encoding='utf-8'))
    for d in data:
        idx = int(d.get('idx'))
        title = str(d.get('title') or '')
        text = str(d.get('text') or '')
        yield {
            'idx': idx,
            'title': title,
            'paragraph_text': text,
            'corpus_idx': idx + offset,
            'is_supporting': False,
            'local_idx': idx,
        }


def _iter_lveval_corpus_paragraphs(corpus_path: Path, offset: int) -> Iterator[dict[str, Any]]:
    data = json.loads(corpus_path.read_text(encoding='utf-8'))
    for d in data:
        idx = int(d.get('idx'))
        title = str(d.get('title') or '')
        text = str(d.get('text') or '')
        yield {
            'idx': idx,
            'title': title,
            'paragraph_text': text,
            'corpus_idx': idx + offset,
            'is_supporting': False,
            'local_idx': idx,
        }


def _extract_unique_musique_paragraphs(musique_qa_path: Path, offset: int) -> list[dict[str, Any]]:
    data = json.loads(musique_qa_path.read_text(encoding='utf-8'))
    by_idx: dict[int, dict[str, Any]] = {}
    for item in data:
        paras = item.get('paragraphs')
        if not isinstance(paras, list):
            continue
        for p in paras:
            if not isinstance(p, dict):
                continue
            cidx = p.get('corpus_idx')
            if cidx is None:
                continue
            try:
                idx = int(cidx)
            except Exception:
                continue
            if idx in by_idx:
                continue
            title = str(p.get('title') or '')
            text = str(p.get('paragraph_text') or p.get('text') or '').strip()
            if not text:
                continue
            by_idx[idx] = {
                'idx': idx,
                'title': title,
                'paragraph_text': text,
                'corpus_idx': idx + offset,
                'is_supporting': False,
                'local_idx': idx,
            }
    # Return ordered for deterministic output
    out = [by_idx[k] for k in sorted(by_idx.keys())]
    return out


def _write_combined_qa_plus_corpus(
    out_path: Path,
    two_wiki_qa_path: Path,
    offsets: dict[str, int],
    hotpot_corpus_path: Path,
    musique_qa_path: Path,
    lveval_corpus_path: Path,
    chunk_size: int,
    rebuild: bool,
) -> None:
    if out_path.exists() and not rebuild:
        return

    two_wiki = json.loads(two_wiki_qa_path.read_text(encoding='utf-8'))
    if not isinstance(two_wiki, list) or not two_wiki:
        raise RuntimeError(f"2wiki QA file not a non-empty list: {two_wiki_qa_path}")

    # We stream-write a JSON list to avoid holding the full combined file in memory.
    with out_path.open('w', encoding='utf-8') as f:
        f.write('[\n')
        first_item = True

        def write_item(it: Any):
            nonlocal first_item
            if not first_item:
                f.write(',\n')
            first_item = False
            json.dump(it, f, ensure_ascii=False)

        # 1) Real 2wiki QA items first (evaluation set)
        for it in two_wiki:
            write_item(it)

        # 2) Hotpot corpus as paragraphs (offset applied)
        hot_paras = list(_iter_hotpot_corpus_paragraphs(hotpot_corpus_path, int(offsets['hotpot'])))
        for ci, chunk in enumerate(_chunks(hot_paras, int(chunk_size))):
            write_item({'id': f'hotpot_corpus_chunk_{ci}', 'paragraphs': chunk})

        # 3) MuSiQue unique paragraphs (offset applied)
        mus_paras = _extract_unique_musique_paragraphs(musique_qa_path, int(offsets['musique']))
        for ci, chunk in enumerate(_chunks(mus_paras, int(chunk_size))):
            write_item({'id': f'musique_corpus_chunk_{ci}', 'paragraphs': chunk})

        # 4) LVEVAL corpus as paragraphs (offset applied)
        lve_paras = list(_iter_lveval_corpus_paragraphs(lveval_corpus_path, int(offsets['lveval'])))
        for ci, chunk in enumerate(_chunks(lve_paras, int(chunk_size))):
            write_item({'id': f'lveval_corpus_chunk_{ci}', 'paragraphs': chunk})

        f.write('\n]\n')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build combined v5 artifacts for large-scale robustness')
    p.add_argument('--out_dir', type=str, default='RobustnessCombined')
    p.add_argument('--rebuild', action='store_true')
    p.add_argument('--chunk_size', type=int, default=200)

    # Inputs
    p.add_argument('--two_wiki_qa', type=str, default='2WikiMultihopQA/2wikimultihopqa.json')
    p.add_argument('--hotpot_corpus', type=str, default='HotpotQA/hotpotqa_corpus.json')
    p.add_argument('--musique_qa', type=str, default='MuSiQue/musique.json')
    p.add_argument('--lveval_corpus', type=str, default='LVEVAL/lveval_corpus.json')

    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    out_dir = (root / str(args.out_dir)).resolve()
    _ensure_dir(out_dir)

    datasets: dict[str, DatasetPaths] = {
        '2wiki': DatasetPaths(
            name='2wiki',
            db_path=root / '2WikiMultihopQA' / 'metadata_v5.db',
            bm25_dir=root / '2WikiMultihopQA' / 'bm25_index_v5',
            path_embeddings_npz=root / '2WikiMultihopQA' / 'path_embeddings_v5.npz',
            corpus_kind='2wiki_qa',
            corpus_path=root / str(args.two_wiki_qa),
        ),
        'hotpot': DatasetPaths(
            name='hotpot',
            db_path=root / 'HotpotQA' / 'metadata_v5.db',
            bm25_dir=root / 'HotpotQA' / 'bm25_index_v5',
            path_embeddings_npz=root / 'HotpotQA' / 'path_embeddings_v5.npz',
            corpus_kind='hotpot_corpus',
            corpus_path=root / str(args.hotpot_corpus),
        ),
        'musique': DatasetPaths(
            name='musique',
            db_path=root / 'MuSiQue' / 'metadata_v5.db',
            bm25_dir=root / 'MuSiQue' / 'bm25_index_v5',
            path_embeddings_npz=root / 'MuSiQue' / 'path_embeddings_v5.npz',
            corpus_kind='musique_qa',
            corpus_path=root / str(args.musique_qa),
        ),
        'lveval': DatasetPaths(
            name='lveval',
            db_path=root / 'LVEVAL' / 'metadata_v5.db',
            bm25_dir=root / 'LVEVAL' / 'bm25_index_v5',
            path_embeddings_npz=root / 'LVEVAL' / 'path_embeddings_v5.npz',
            corpus_kind='lveval_corpus',
            corpus_path=root / str(args.lveval_corpus),
        ),
    }

    # Validate inputs exist
    for k, ds in datasets.items():
        for pth in [ds.db_path, ds.bm25_dir, ds.path_embeddings_npz]:
            if not pth.exists():
                raise SystemExit(f"Missing required artifact for {k}: {pth}")

    offsets = _build_offsets(datasets)
    (out_dir / 'offsets.json').write_text(json.dumps(offsets, indent=2), encoding='utf-8')

    # 1) merged DB
    _merge_metadata_db(out_dir / 'metadata_v5.db', datasets, offsets, rebuild=bool(args.rebuild))

    # 2) merged embeddings npz (with doc_id offsets)
    meta_arrays = _merge_path_embeddings(out_dir / 'path_embeddings_v5.npz', datasets, offsets, rebuild=bool(args.rebuild))

    # 3) merged BM25 index (rebuilt locally; no LLM)
    _build_bm25_index(out_dir / 'bm25_index_v5', meta_arrays, rebuild=bool(args.rebuild))

    # 4) combined QA+corpus file for passage lookup
    _write_combined_qa_plus_corpus(
        out_path=out_dir / '2wiki_qa_plus_4corpus.json',
        two_wiki_qa_path=root / str(args.two_wiki_qa),
        offsets=offsets,
        hotpot_corpus_path=root / str(args.hotpot_corpus),
        musique_qa_path=root / str(args.musique_qa),
        lveval_corpus_path=root / str(args.lveval_corpus),
        chunk_size=int(args.chunk_size),
        rebuild=bool(args.rebuild),
    )

    print('[OK] Built combined artifacts under:', out_dir)
    print('  -', out_dir / 'metadata_v5.db')
    print('  -', out_dir / 'bm25_index_v5')
    print('  -', out_dir / 'path_embeddings_v5.npz')
    print('  -', out_dir / '2wiki_qa_plus_4corpus.json')


if __name__ == '__main__':
    main()
