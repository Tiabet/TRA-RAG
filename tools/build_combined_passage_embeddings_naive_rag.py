#!/usr/bin/env python3
"""Build combined passage embedding cache for NaiveRAG cosine retriever.

Why this exists
- Your v12 pipeline uses *path* indices (metadata DB + bm25_index_v5 + path_embeddings_v5.npz).
- NaiveRAG cosine retriever uses a *passage* embedding cache (.npz) produced from a dataset file.
- For the robustness test, NaiveRAG must retrieve from the same *combined* corpus.

Strategy
- Reuse existing passage embedding caches where available:
  - 2WikiMultihopQA/passage_embeddings_naive_rag.npz (6119)
  - HotpotQA/passage_embeddings_naive_rag.npz (9811)
  - MuSiQue/passage_embeddings_naive_rag.npz (11656)
- LVEVAL's existing cache in this repo is small (built from QA distractors), so we embed LVEVAL corpus
  from LVEVAL/lveval_corpus.json and assign numeric doc_ids with the same offset scheme.

Output
- <out_dir>/passage_embeddings_naive_rag.npz

This script does NOT modify any pipeline logic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dotenv import load_dotenv

# Ensure repo root is importable when running from tools/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from llm_provider import create_async_embed_client, detect_provider


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build combined NaiveRAG passage embedding cache')
    p.add_argument('--out_dir', type=str, default='RobustnessCombined')
    p.add_argument('--offsets', type=str, default='RobustnessCombined/offsets.json')
    p.add_argument('--rebuild', action='store_true')

    p.add_argument('--two_wiki_cache', type=str, default='2WikiMultihopQA/passage_embeddings_naive_rag.npz')
    p.add_argument('--hotpot_cache', type=str, default='HotpotQA/passage_embeddings_naive_rag.npz')
    p.add_argument('--musique_cache', type=str, default='MuSiQue/passage_embeddings_naive_rag.npz')

    p.add_argument('--lveval_corpus', type=str, default='LVEVAL/lveval_corpus.json')
    p.add_argument('--lveval_limit', type=int, default=0, help='0=all, otherwise only first N LVEVAL docs')
    p.add_argument('--embed_batch_size', type=int, default=100)
    p.add_argument('--embed_concurrency', type=int, default=10)

    return p.parse_args()


def _load_cache(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    data = np.load(str(path), allow_pickle=True)
    emb = data['embeddings'].astype(np.float32, copy=False)
    titles = [str(x) for x in list(data.get('titles', []))]
    doc_ids = [str(x) for x in list(data.get('doc_ids', []))]
    if len(titles) != emb.shape[0] or len(doc_ids) != emb.shape[0]:
        raise RuntimeError(f"Cache length mismatch: {path} (emb={emb.shape}, titles={len(titles)}, doc_ids={len(doc_ids)})")
    return emb, titles, doc_ids


async def _embed_texts(texts: list[str], *, batch_size: int, concurrency: int) -> np.ndarray:
    cfg = detect_provider()
    client = create_async_embed_client(cfg)
    model = cfg.embed_model

    sem = asyncio.Semaphore(int(concurrency))

    async def one_batch(batch: list[str]) -> list[list[float]]:
        async with sem:
            resp = await client.embeddings.create(model=model, input=[t.replace('\n', ' ') for t in batch])
            return [d.embedding for d in resp.data]

    tasks = []
    for i in range(0, len(texts), int(batch_size)):
        tasks.append(one_batch(texts[i:i + int(batch_size)]))

    out: list[list[float]] = []
    for coro in asyncio.as_completed(tasks):
        out.extend(await coro)

    return np.asarray(out, dtype=np.float32)


async def main_async() -> None:
    load_dotenv()
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    out_dir = (root / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'passage_embeddings_naive_rag.npz'

    if out_path.exists() and not bool(args.rebuild):
        print('[Skip] combined passage cache exists:', out_path)
        return

    offsets = json.loads((root / str(args.offsets)).read_text(encoding='utf-8'))

    emb_parts: list[np.ndarray] = []
    titles_all: list[str] = []
    doc_ids_all: list[str] = []

    # 1) Reuse 2wiki/hotpot/musique caches with numeric doc_id offsets
    for key, cache_path in [
        ('2wiki', Path(root / str(args.two_wiki_cache))),
        ('hotpot', Path(root / str(args.hotpot_cache))),
        ('musique', Path(root / str(args.musique_cache))),
    ]:
        off = int(offsets[key])
        emb, titles, doc_ids = _load_cache(cache_path)
        # Validate numeric IDs
        new_doc_ids = []
        for d in doc_ids:
            try:
                new_doc_ids.append(str(int(d) + off))
            except Exception:
                raise RuntimeError(f"Non-numeric doc_id in {cache_path}: {d}")

        emb_parts.append(emb)
        titles_all.extend(titles)
        doc_ids_all.extend(new_doc_ids)
        print(f"[OK] Loaded {key} cache: n={emb.shape[0]} offset={off}")

    # 2) Embed LVEVAL corpus (numeric idx) and offset doc_ids
    lve_off = int(offsets['lveval'])
    lve_data = json.loads((root / str(args.lveval_corpus)).read_text(encoding='utf-8'))
    if not isinstance(lve_data, list):
        raise RuntimeError('LVEVAL corpus must be a list')

    if int(args.lveval_limit) > 0:
        lve_data = lve_data[: int(args.lveval_limit)]

    lve_titles: list[str] = []
    lve_doc_ids: list[str] = []
    lve_texts: list[str] = []
    for d in lve_data:
        if not isinstance(d, dict):
            continue
        idx = d.get('idx')
        if idx is None:
            continue
        try:
            idx_i = int(idx)
        except Exception:
            continue
        title = str(d.get('title') or '')
        text = str(d.get('text') or '')
        lve_titles.append(title)
        lve_doc_ids.append(str(idx_i + lve_off))
        lve_texts.append(f"{title}\n{text}")

    if lve_texts:
        print(f"[EMB] Embedding LVEVAL corpus: n={len(lve_texts)} (offset={lve_off})")
        lve_emb = await _embed_texts(lve_texts, batch_size=int(args.embed_batch_size), concurrency=int(args.embed_concurrency))
        emb_parts.append(lve_emb)
        titles_all.extend(lve_titles)
        doc_ids_all.extend(lve_doc_ids)

    # 3) Write combined NPZ
    if not emb_parts:
        raise RuntimeError('No embeddings were collected')

    emb_all = np.concatenate(emb_parts, axis=0).astype(np.float32, copy=False)
    if len(titles_all) != emb_all.shape[0] or len(doc_ids_all) != emb_all.shape[0]:
        raise RuntimeError('Combined lengths mismatch')

    np.savez(
        str(out_path),
        embeddings=emb_all,
        titles=np.asarray(titles_all, dtype=object),
        doc_ids=np.asarray(doc_ids_all, dtype=object),
    )

    print('[OK] Wrote combined passage cache:', out_path)
    print('     shape:', emb_all.shape)


def main() -> None:
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
