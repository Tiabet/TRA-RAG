#!/usr/bin/env python3
"""Run 2Wiki serialization and retrieval-budget sensitivity on cached QD queries."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from bm25_indexer import BM25Indexer
from evaluate_retrieval import get_gold_corpus_doc_ids
from evaluate_mrqa import compute_metrics_with_aliases
from hybrid_path_retriever import HybridPathRetriever
from llm_provider import create_async_chat_client, detect_provider
from new_multihop_pipeline_paths_hint_expansion import NewMultihopPipelineV12PathsHintExpansion
from path_embedding_generator import PathEmbeddingGenerator
from query_decomposition import QueryDecomposition, SubQuestion


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "2WikiMultihopQA"
DEFAULT_RESULTS = ROOT / "Results" / "reviewer_sensitivity_2wiki100"
VARIANTS = ("natural", "key_value", "json", "triple")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _value_for_json(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return raw


def serialize_entry(entry: dict[str, Any], variant: str) -> str:
    title = str(entry.get("entity_title") or entry.get("title") or "")
    path = str(entry.get("key_path") or "")
    value = entry.get("value", "")
    if variant == "natural":
        return str(entry.get("text") or "")
    if variant == "key_value":
        return f"[Entity: {title}] {path} = {value}"
    if variant == "json":
        return json.dumps(
            {"entity": title, "path": path, "value": _value_for_json(value)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    if variant == "triple":
        return f"({title}, {path}, {value})"
    raise ValueError(f"Unknown serialization variant: {variant}")


def prepare_variant_texts() -> dict[str, Path]:
    source = DATA / "embedding_texts_v5.json"
    rows = _read_json(source)
    out: dict[str, Path] = {"natural": source}
    for variant in VARIANTS[1:]:
        target = DATA / f"embedding_texts_v5_{variant}.json"
        transformed = []
        for row in rows:
            item = dict(row)
            item["text"] = serialize_entry(item, variant)
            transformed.append(item)
        _write_json(target, transformed)
        out[variant] = target
        print(f"[OK] {variant}: {len(transformed)} facts -> {target}")
    return out


async def prepare_serialization_indexes(batch_size: int, concurrency: int) -> None:
    text_paths = prepare_variant_texts()

    async def embed_one(variant: str) -> None:
        output = DATA / f"path_embeddings_v5_{variant}.npz"
        if output.exists():
            print(f"[SKIP] Existing embedding index: {output}")
            return
        generator = PathEmbeddingGenerator(batch_size=batch_size, max_concurrency=concurrency)
        await generator.generate_embeddings(str(text_paths[variant]), str(output))

    await asyncio.gather(*(embed_one(v) for v in VARIANTS))

    async def bm25_one(variant: str) -> None:
        output = DATA / f"bm25_index_v5_{variant}"
        if (output / "metadata.json").exists():
            print(f"[SKIP] Existing BM25 index: {output}")
            return
        indexer = BM25Indexer(use_stemming=True)
        await asyncio.to_thread(
            indexer.build_index,
            str(text_paths[variant]),
            str(output),
            True,
            False,
        )

    await asyncio.gather(*(bm25_one(v) for v in VARIANTS))


def _load_subset(limit: int, cache_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gold = _read_json(DATA / "2wikimultihopqa.json")[:limit]
    cached = _read_json(cache_path).get("results", [])
    by_id = {str(r.get("id")): r for r in cached if isinstance(r, dict)}
    selected = []
    for i, item in enumerate(gold):
        qid = str(item.get("_id") or item.get("id") or f"q{i}")
        row = by_id.get(qid)
        if not row or not (row.get("decomposition") or {}).get("subquestions"):
            raise RuntimeError(f"Missing cached QD decomposition for {qid}")
        selected.append(row)
    return gold, selected


def _cached_questions(cached: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    seen = set()
    for row in cached:
        for sq in (row.get("decomposition") or {}).get("subquestions", []):
            q = str(sq.get("actual_question") or sq.get("question") or "").strip()
            if q and q not in seen:
                seen.add(q)
                values.append(q)
    return values


async def build_query_embedding_cache(questions: list[str], batch_size: int, concurrency: int) -> dict[str, np.ndarray]:
    generator = PathEmbeddingGenerator(batch_size=batch_size, max_concurrency=concurrency)
    matrix = await generator.embed_texts(questions)
    normalized: dict[str, np.ndarray] = {}
    for q, vector in zip(questions, matrix):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        normalized[q] = vector / norm if norm > 0 else vector
    return normalized


def _build_decomposition(row: dict[str, Any]) -> QueryDecomposition:
    raw = row.get("decomposition") or {}
    subquestions = []
    for item in raw.get("subquestions", []):
        sq = SubQuestion(
            id=str(item.get("id") or f"SQ{len(subquestions) + 1}"),
            question=str(item.get("question") or ""),
            depends_on=[str(x) for x in (item.get("depends_on") or [])],
            reasoning="cached",
        )
        sq.actual_question = str(item.get("actual_question") or item.get("question") or "")
        sq.answer = item.get("answer")
        sq.support_paths_for_passages = []
        subquestions.append(sq)
    return QueryDecomposition(
        main_query=str(raw.get("main_query") or row.get("question") or ""),
        question_type="cached",
        reasoning="cached",
        subquestions=subquestions,
    )


def _attach_support_paths(sq: SubQuestion, passages: list[dict[str, Any]]) -> None:
    support = []
    for p in passages:
        doc_id = p.get("support_path_doc_id")
        key = p.get("support_path_key_path")
        value = p.get("support_path_value")
        if doc_id is None or key is None or value is None:
            continue
        support.append(
            {
                "index": p.get("support_path_index"),
                "title": p.get("support_path_title") or p.get("title"),
                "doc_id": str(doc_id),
                "source_title": p.get("support_path_source_title") or p.get("source_title"),
                "entity_title": p.get("support_path_entity_title") or p.get("entity_title"),
                "key_path": str(key),
                "value": str(value),
                "score": p.get("support_path_score", p.get("passage_score", p.get("score"))),
                "origin": "passage_support",
            }
        )
    sq.support_paths_for_passages = support


def _artifact_paths(variant: str) -> tuple[Path, Path]:
    return DATA / f"bm25_index_v5_{variant}", DATA / f"path_embeddings_v5_{variant}.npz"


async def run_config(
    name: str,
    variant: str,
    rrf_k: int,
    sq_path_pool: int,
    final_rerank_passages: int,
    gold: list[dict[str, Any]],
    cached: list[dict[str, Any]],
    query_embeddings: dict[str, np.ndarray],
    concurrency: int,
    output_dir: Path,
    client: Any | None = None,
    chat_model: str = "",
) -> dict[str, Any]:
    bm25_path, embedding_path = _artifact_paths(variant)
    retriever = HybridPathRetriever(
        bm25_index_path=str(bm25_path),
        embeddings_path=str(embedding_path),
        bm25_weight=1.0,
        dense_weight=1.3,
        rrf_k=rrf_k,
    )

    async def cached_embed(query: str) -> np.ndarray:
        vector = query_embeddings.get(str(query).strip())
        if vector is None:
            raise KeyError(f"Query embedding was not cached: {query}")
        return vector

    retriever.embed_query = cached_embed  # type: ignore[method-assign]
    pipeline = NewMultihopPipelineV12PathsHintExpansion(
        client=client,
        retriever=retriever,
        hotpotqa_path=str(DATA / "2wikimultihopqa.json"),
        db_path=str(DATA / "metadata_v5.db"),
        top_k_passages=20,
        top_k_paths=sq_path_pool,
        answer_k_passages=5,
        answer_k_paths=30,
        path_fetch_k=max(100, sq_path_pool * 10),
        verbose=False,
        seed_k=20,
        expansion_k=10,
        expansion_dense_candidates=500,
        seed_passages_in_final=3,
        sq_fusion_method="rrf",
        final_selection_mode="rerank",
        chat_model=chat_model,
    )
    seed_passages = 5 - int(final_rerank_passages)
    pipeline._PASSAGE_SEED_RERANK = {**pipeline._PASSAGE_SEED_RERANK, 5: (seed_passages, final_rerank_passages)}

    sem = asyncio.Semaphore(max(1, concurrency))

    async def process_one(i: int) -> dict[str, Any]:
        async with sem:
            started = time.perf_counter()
            decomposition = _build_decomposition(cached[i])
            for sq in decomposition.subquestions:
                passages, paths = await pipeline.retrieve_for_query(sq.actual_question)
                sq.retrieved_passages = passages
                sq.retrieved_paths = paths
                _attach_support_paths(sq, passages)
            final_paths, final_passages = await pipeline._select_top_paths_and_passages_from_decomposition(
                decomposition,
                top_paths_k=30,
                top_passages_k=5,
            )
            predicted_answer = ""
            qa_error = None
            metrics = None
            if client is not None:
                decomposition._final_selected_paths = final_paths
                decomposition._final_selected_passages = final_passages
                try:
                    predicted_answer = await pipeline.generate_final_answer(
                        str(gold[i].get("question") or ""), decomposition, final_passages
                    )
                    aliases = [gold[i].get("gold_answer") or gold[i].get("answer")]
                    aliases.extend(gold[i].get("answer_aliases") or [])
                    metrics = compute_metrics_with_aliases(
                        [str(x) for x in aliases if x is not None and str(x).strip()],
                        predicted_answer,
                    )
                except Exception as exc:
                    qa_error = f"{type(exc).__name__}: {exc}"
            retrieved = [str(p.get("doc_id")) for p in final_passages if p.get("doc_id") is not None]
            gold_ids, missing, ambiguous = get_gold_corpus_doc_ids(gold[i])
            hits = set(retrieved) & gold_ids
            recall = (len(hits) / len(gold_ids)) if gold_ids else 0.0
            return {
                "id": str(gold[i].get("_id") or gold[i].get("id") or f"q{i}"),
                "gold_doc_ids": sorted(gold_ids),
                "retrieved_doc_ids": retrieved,
                "recall_at_5": recall,
                "hit_at_5": int(bool(hits)),
                "hit_all_at_5": int(bool(gold_ids) and gold_ids.issubset(set(retrieved))),
                "missing_gold_titles": sorted(missing),
                "ambiguous_gold_titles": sorted(ambiguous),
                "elapsed_s": time.perf_counter() - started,
                "num_subquestions": len(decomposition.subquestions),
                "num_paths": len(final_paths),
                "gold_answer": gold[i].get("gold_answer") or gold[i].get("answer"),
                "answer_aliases": gold[i].get("answer_aliases") or [],
                "predicted_answer": predicted_answer,
                "qa_metrics": metrics,
                "qa_error": qa_error,
            }

    started = time.perf_counter()
    rows = await asyncio.gather(*(process_one(i) for i in range(len(gold))))
    elapsed = time.perf_counter() - started
    summary = {
        "name": name,
        "serialization": variant,
        "rrf_k": rrf_k,
        "sq_path_pool": sq_path_pool,
        "effective_path_fetch_k": max(100, sq_path_pool * 10, 200),
        "final_rerank_passages": final_rerank_passages,
        "final_seed_passages": seed_passages,
        "n": len(rows),
        "recall_at_5": statistics.mean(r["recall_at_5"] for r in rows),
        "hit_at_5": statistics.mean(r["hit_at_5"] for r in rows),
        "hit_all_at_5": statistics.mean(r["hit_all_at_5"] for r in rows),
        "mean_latency_s": statistics.mean(r["elapsed_s"] for r in rows),
        "wall_time_s": elapsed,
    }
    qa_rows = [r for r in rows if isinstance(r.get("qa_metrics"), dict)]
    if client is not None:
        summary.update(
            {
                "qa_success": len(qa_rows),
                "qa_failures": len(rows) - len(qa_rows),
                "mrqa_f1": statistics.mean(r["qa_metrics"]["f1"] for r in qa_rows) if qa_rows else 0.0,
                "accuracy": statistics.mean(r["qa_metrics"]["accuracy"] for r in qa_rows) if qa_rows else 0.0,
                "exact_match": statistics.mean(r["qa_metrics"]["exact_match"] for r in qa_rows) if qa_rows else 0.0,
            }
        )
    pipeline.close()
    del pipeline, retriever
    output = {"summary": summary, "results": rows}
    _write_json(output_dir / f"{name}.json", output)
    print("[RESULT] " + json.dumps(summary, ensure_ascii=False))
    return summary


async def run_sensitivity(
    limit: int,
    concurrency: int,
    embed_batch_size: int,
    embed_concurrency: int,
    run_qa: bool,
    cache_path: Path,
    output_dir: Path,
    resume: bool,
    selected_configs: set[str] | None,
) -> None:
    gold, cached = _load_subset(limit, cache_path)
    questions = _cached_questions(cached)
    print(f"[INFO] Cached QD: {len(gold)} questions, {len(questions)} unique subquestions")
    query_embeddings = await build_query_embedding_cache(questions, embed_batch_size, embed_concurrency)
    client = None
    chat_model = ""
    if run_qa:
        cfg = detect_provider()
        client = create_async_chat_client(cfg)
        chat_model = cfg.chat_model
        print(f"[INFO] QA provider={cfg.provider}, model={chat_model}")

    configs = []
    for variant in VARIANTS:
        configs.append((f"serialization_{variant}", variant, 60, 50, 2, "serialization"))
    for k in (10, 100):
        configs.append((f"rrf_k{k}", "natural", k, 50, 2, "budget"))
    for pool in (20, 100):
        configs.append((f"sq_path_pool{pool}", "natural", 60, pool, 2, "budget"))
    for rerank in (0, 4):
        configs.append((f"final_rerank_passages{rerank}", "natural", 60, 50, rerank, "budget"))
    if selected_configs:
        configs = [config for config in configs if config[0] in selected_configs]
        unknown = selected_configs - {config[0] for config in configs}
        if unknown:
            raise ValueError(f"Unknown config names: {sorted(unknown)}")

    summaries = []
    try:
        for name, variant, rrf_k, pool, rerank, group in configs:
            output_path = output_dir / f"{name}.json"
            if resume and output_path.exists():
                existing = _read_json(output_path).get("summary", {})
                qa_complete = (not run_qa) or (
                    int(existing.get("qa_success") or 0) == limit
                    and int(existing.get("qa_failures") or 0) == 0
                )
                if int(existing.get("n") or 0) == limit and qa_complete:
                    existing["group"] = group
                    summaries.append(existing)
                    print(f"[SKIP] Complete config: {name}")
                    continue
            summary = await run_config(
                name,
                variant,
                rrf_k,
                pool,
                rerank,
                gold,
                cached,
                query_embeddings,
                concurrency,
                output_dir,
                client,
                chat_model,
            )
            summary["group"] = group
            summaries.append(summary)

        payload = {
            "dataset": "2wiki",
            "limit": limit,
            "subset_sha1": hashlib.sha1("\n".join(str(x.get("_id") or x.get("id")) for x in gold).encode()).hexdigest(),
            "concurrency": concurrency,
            "cached_qd_source": str(cache_path.relative_to(ROOT) if cache_path.is_relative_to(ROOT) else cache_path),
            "qa_enabled": run_qa,
            "chat_model": chat_model or None,
            "summaries": summaries,
        }
        _write_json(output_dir / "summary.json", payload)
    finally:
        if client is not None:
            await client.close()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--embed_batch_size", type=int, default=512)
    parser.add_argument("--embed_concurrency", type=int, default=40)
    parser.add_argument("--qa", action="store_true")
    parser.add_argument(
        "--cached_qd_source",
        type=Path,
        default=ROOT / "Results" / "grid_2wiki_v12_rerun" / "2wiki_v12_p5_path30.json",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--configs",
        default="",
        help="Comma-separated config names; empty runs the complete sensitivity suite.",
    )
    args = parser.parse_args()
    load_dotenv(override=False)
    if not args.prepare and not args.run:
        args.prepare = args.run = True
    if args.prepare:
        await prepare_serialization_indexes(args.embed_batch_size, args.embed_concurrency)
    if args.run:
        await run_sensitivity(
            args.limit,
            args.concurrency,
            args.embed_batch_size,
            args.embed_concurrency,
            args.qa,
            args.cached_qd_source.resolve(),
            args.output_dir.resolve(),
            args.resume,
            {x.strip() for x in args.configs.split(",") if x.strip()} or None,
        )


if __name__ == "__main__":
    asyncio.run(main())
