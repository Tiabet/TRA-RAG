#!/usr/bin/env python3
"""Test Pipeline v10 (no QD, value-only 1-hop expansion) on 200 questions."""

import asyncio
import json
import time
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

from hybrid_path_retriever import HybridPathRetriever
from new_pipeline_v10 import PipelineV10, MetadataLinkerValuesOnly


CONCURRENCY = 120

DATASET_CONFIGS = {
    "hotpotqa": {
        "data_path": "HotpotQA/hotpotqa_sample_200.json",
        "metadata_json": "HotpotQA/hotpotqa_sample_200_metadata.json",
        "bm25_index": "HotpotQA/bm25_index",
        "path_embeddings": "HotpotQA/path_embeddings.npz",
        "result_path": "Results/test_hotpot_v10_200_results_v1.json",
    },
    "musique": {
        "data_path": "MuSiQue/musique_sample_200.json",
        "metadata_json": "MuSiQue/musique_sample_200_metadata.json",
        "bm25_index": "MuSiQue/bm25_index",
        "path_embeddings": "MuSiQue/path_embeddings.npz",
        "result_path": "Results/test_musique_v10_200_results_v1.json",
    },
}


async def process_one(pipeline, item, idx, total):
    q = item["question"]
    gold = item.get("answer")
    qid = item.get("_id", f"q{idx}")

    start = time.time()
    try:
        out = await pipeline.run(q)
        elapsed = time.time() - start
        pred = out.get("predicted_answer")
        num = len(out.get("retrieved_passages", []) or [])
        print(f"[{idx+1:3d}/{total}] ✓ ({elapsed:.1f}s, {num}p) {q[:60]}...")
        if gold is not None:
            print(f"           Gold: {gold}")
        print(f"           Pred: {pred}")

        return {
            "id": qid,
            "question": q,
            "gold_answer": gold,
            "predicted_answer": pred,
            "answer_aliases": item.get("answer_aliases", []),
            "time": elapsed,
            "num_passages": num,
            "success": True,
            "retrieved_passages": out.get("retrieved_passages"),
            "retrieval_info": out.get("retrieval_info"),
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"[{idx+1:3d}/{total}] ✗ ({elapsed:.1f}s) {str(e)[:120]}...")
        return {"id": qid, "question": q, "gold_answer": gold, "predicted_answer": None, "time": elapsed, "success": False, "error": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Test pipeline v10")
    parser.add_argument("--dataset", type=str, default="musique", choices=["hotpotqa", "musique"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dense_weight", type=float, default=0.6)
    parser.add_argument("--initial_top_paths", type=int, default=10)
    parser.add_argument("--final_top_passages", type=int, default=5)
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("ALICE_OPENAI_KEY"):
        print("Error: ALICE_OPENAI_KEY not found")
        return

    cfg = DATASET_CONFIGS[args.dataset]

    client = AsyncOpenAI(api_key=os.getenv("ALICE_OPENAI_KEY"), base_url=os.getenv("ALICE_CHAT_URL"))

    retriever = HybridPathRetriever(
        bm25_index_path=cfg["bm25_index"],
        embeddings_path=cfg["path_embeddings"],
        dense_weight=args.dense_weight,
        bm25_weight=1.0 - args.dense_weight,
    )

    linker = MetadataLinkerValuesOnly(cfg["metadata_json"])
    pipeline = PipelineV10(
        client=client,
        retriever=retriever,
        linker=linker,
        dataset_path=cfg["data_path"],
        initial_top_paths=args.initial_top_paths,
        final_top_passages=args.final_top_passages,
        verbose=False,
    )

    with open(cfg["data_path"], "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.limit is not None:
        data = data[args.offset : args.offset + args.limit]
    else:
        data = data[args.offset :]

    sem = asyncio.Semaphore(CONCURRENCY)

    async def bound(item, i):
        async with sem:
            return await process_one(pipeline, item, i, len(data))

    start = time.time()
    results = await asyncio.gather(*[bound(item, i) for i, item in enumerate(data)])
    total = time.time() - start

    ok = [r for r in results if r.get("success")]
    avg_time = total / len(data) if data else 0
    avg_passages = (sum(r.get("num_passages", 0) for r in ok) / len(ok)) if ok else 0

    output = {
        "config": {
            "pipeline_version": "v10",
            "dataset": args.dataset,
            "concurrency": CONCURRENCY,
            "dense_weight": args.dense_weight,
            "bm25_weight": 1.0 - args.dense_weight,
            "initial_top_paths": args.initial_top_paths,
            "final_top_passages": args.final_top_passages,
            "linking": "value-only (no keys, no filters)",
        },
        "summary": {
            "total_questions": len(data),
            "successful": len(ok),
            "errors": len(data) - len(ok),
            "total_time": total,
            "avg_time_per_question": avg_time,
            "avg_passages": avg_passages,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(cfg["result_path"]), exist_ok=True)
    with open(cfg["result_path"], "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved: {cfg['result_path']}")


if __name__ == "__main__":
    asyncio.run(main())
