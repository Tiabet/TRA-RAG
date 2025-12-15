#!/usr/bin/env python3
"""Test New Multi-hop Pipeline v8 on 200 Questions.

v8 behavior:
- Independent SQs: global hybrid retrieval top-3 titles
- Dependent SQs: restrict retrieval to dependency expansion pool (path indices) and RRF within that pool
- Pool updates after every SQ (expand from top passages used)
"""

import asyncio
import json
import time
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

from new_multihop_pipeline_v8 import NewMultihopPipelineV8
from new_multihop_pipeline_v7 import MetadataLinkerV7
from hybrid_path_retriever import HybridPathRetriever


CONCURRENCY = 80

DATASET_CONFIGS = {
    "hotpotqa": {
        "data_path": "HotpotQA/hotpotqa_sample_200.json",
        "metadata_json": "HotpotQA/hotpotqa_sample_200_metadata.json",
        "bm25_index": "HotpotQA/bm25_index",
        "path_embeddings": "HotpotQA/path_embeddings.npz",
        "result_path": "Results/test_hotpot_v8_200_results_v1.json",
    },
    "musique": {
        "data_path": "MuSiQue/musique_sample_200.json",
        "metadata_json": "MuSiQue/musique_sample_200_metadata.json",
        "bm25_index": "MuSiQue/bm25_index",
        "path_embeddings": "MuSiQue/path_embeddings.npz",
        "result_path": "Results/test_musique_v8_200_results_v1.json",
    },
}


async def process_single_question(pipeline, item, idx, total):
    question = item["question"]
    gold_answer = item.get("answer")
    qid = item.get("_id", f"q{idx}")

    start = time.time()
    try:
        result = await pipeline.run(question)
        elapsed = time.time() - start

        predicted = result.get("predicted_answer")
        num_passages = len(result.get("passages", []) or [])

        print(f"[{idx+1:3d}/{total}] ✓ ({elapsed:.1f}s, {num_passages}p) {question[:60]}...")
        if gold_answer is not None:
            print(f"           Gold: {gold_answer}")
        print(f"           Pred: {predicted}")

        return {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "answer_aliases": item.get("answer_aliases", []),
            "time": elapsed,
            "num_passages": num_passages,
            "success": True,
            "decomposition": result.get("decomposition"),
            "passages": result.get("passages"),
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"[{idx+1:3d}/{total}] ✗ ({elapsed:.1f}s) Exception: {str(e)[:120]}...")
        return {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": None,
            "answer_aliases": item.get("answer_aliases", []),
            "time": elapsed,
            "success": False,
            "error": str(e),
        }


async def main():
    parser = argparse.ArgumentParser(description="Test Pipeline v8")
    parser.add_argument("--dataset", type=str, default="musique", choices=["hotpotqa", "musique"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dense_weight", type=float, default=0.6)
    parser.add_argument("--max_docs_per_value", type=int, default=200)
    parser.add_argument("--top_k_fallback", type=int, default=10)
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("ALICE_OPENAI_KEY"):
        print("Error: ALICE_OPENAI_KEY not found in .env")
        return

    config = DATASET_CONFIGS[args.dataset]
    chat_client = AsyncOpenAI(api_key=os.getenv("ALICE_OPENAI_KEY"), base_url=os.getenv("ALICE_CHAT_URL"))

    retriever = HybridPathRetriever(
        bm25_index_path=config["bm25_index"],
        embeddings_path=config["path_embeddings"],
        dense_weight=args.dense_weight,
        bm25_weight=1.0 - args.dense_weight,
    )

    linker = MetadataLinkerV7(config["metadata_json"], max_docs_per_value=args.max_docs_per_value)

    pipeline = NewMultihopPipelineV8(
        client=chat_client,
        retriever=retriever,
        linker=linker,
        hotpotqa_path=config["data_path"],
        top_k=3,
        verbose=False,
        top_k_fallback=args.top_k_fallback,
    )

    with open(config["data_path"], "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.limit is not None:
        data = data[args.offset : args.offset + args.limit]
    else:
        data = data[args.offset :]

    sem = asyncio.Semaphore(CONCURRENCY)

    async def bound_process(item, idx):
        async with sem:
            return await process_single_question(pipeline, item, idx, len(data))

    start_time = time.time()
    tasks = [bound_process(item, i) for i, item in enumerate(data)]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time

    successful = [r for r in results if r.get("success")]
    avg_time = total_time / len(data) if data else 0
    avg_passages = (sum(r.get("num_passages", 0) for r in successful) / len(successful)) if successful else 0

    output = {
        "config": {
            "pipeline_version": "v8",
            "dataset": args.dataset,
            "concurrency": CONCURRENCY,
            "top_k": 3,
            "dense_weight": args.dense_weight,
            "bm25_weight": 1.0 - args.dense_weight,
            "max_docs_per_value": args.max_docs_per_value,
            "top_k_fallback": args.top_k_fallback,
        },
        "summary": {
            "total_questions": len(data),
            "successful": len(successful),
            "errors": len(data) - len(successful),
            "total_time": total_time,
            "avg_time_per_question": avg_time,
            "avg_passages_per_question": avg_passages,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(config["result_path"]), exist_ok=True)
    with open(config["result_path"], "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved: {config['result_path']}")


if __name__ == "__main__":
    asyncio.run(main())
