#!/usr/bin/env python3
"""
Test New Multi-hop Pipeline v7 on 200 Questions
==============================================

v7 adds expansion noise filters:
- Exclude relations.relation values from expansion linking
- Degree cutoff: block values with too many linked docs

Usage:
    python test_pipeline_v7_200.py                     # Default: MuSiQue
    python test_pipeline_v7_200.py --dataset hotpotqa  # HotpotQA
    python test_pipeline_v7_200.py --dataset musique   # MuSiQue
"""

import asyncio
import json
import time
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

from new_multihop_pipeline_v7 import NewMultihopPipelineV7, MetadataLinkerV7
from hybrid_path_retriever import HybridPathRetriever


CONCURRENCY = 100

DATASET_CONFIGS = {
    "hotpotqa": {
        "data_path": "HotpotQA/hotpotqa_sample_200.json",
        "metadata_json": "HotpotQA/hotpotqa_sample_200_metadata.json",
        "bm25_index": "HotpotQA/bm25_index",
        "path_embeddings": "HotpotQA/path_embeddings.npz",
        "result_path": "Results/test_hotpot_v7_200_results.json",
    },
    "musique": {
        "data_path": "MuSiQue/musique_sample_200.json",
        "metadata_json": "MuSiQue/musique_sample_200_metadata.json",
        "bm25_index": "MuSiQue/bm25_index",
        "path_embeddings": "MuSiQue/path_embeddings.npz",
        "result_path": "Results/test_musique_v7_200_results_v1.json",
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

        num_passages = 0
        if "decomposition" in result and "subquestions" in result["decomposition"]:
            seen_titles = set()
            for sq in result["decomposition"]["subquestions"]:
                for p in sq.get("retrieved_passages", []) or []:
                    title = p.get("title")
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        num_passages += 1

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
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"[{idx+1:3d}/{total}] ✗ ({elapsed:.1f}s) Exception: {str(e)[:80]}...")
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
    parser = argparse.ArgumentParser(description="Test Pipeline V7")
    parser.add_argument("--dataset", type=str, default="musique", choices=["hotpotqa", "musique"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dense_weight", type=float, default=0.6)
    parser.add_argument("--max_docs_per_value", type=int, default=200, help="Degree cutoff for expansion values")
    args = parser.parse_args()

    load_dotenv()

    if not os.getenv("ALICE_OPENAI_KEY"):
        print("Error: ALICE_OPENAI_KEY not found in .env")
        return

    config = DATASET_CONFIGS[args.dataset]
    print(f"Configuration: {args.dataset.upper()}")
    print(f"- Data: {config['data_path']}")
    print(f"- Metadata: {config['metadata_json']}")
    print(f"- Result: {config['result_path']}")
    print(f"- Dense Weight: {args.dense_weight} (BM25 Weight: {1.0 - args.dense_weight:.2f})")
    print(f"- max_docs_per_value: {args.max_docs_per_value}")

    chat_client = AsyncOpenAI(api_key=os.getenv("ALICE_OPENAI_KEY"), base_url=os.getenv("ALICE_CHAT_URL"))

    print("\nInitializing Components...")
    retriever = HybridPathRetriever(
        bm25_index_path=config["bm25_index"],
        embeddings_path=config["path_embeddings"],
        dense_weight=args.dense_weight,
        bm25_weight=1.0 - args.dense_weight,
    )

    linker = MetadataLinkerV7(config["metadata_json"], max_docs_per_value=args.max_docs_per_value)

    pipeline = NewMultihopPipelineV7(
        client=chat_client,
        retriever=retriever,
        linker=linker,
        hotpotqa_path=config["data_path"],
        top_k=3,
        verbose=False,
    )

    print(f"Loading dataset from {config['data_path']}...")
    with open(config["data_path"], "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.limit is not None:
        data = data[args.offset : args.offset + args.limit]
    else:
        data = data[args.offset :]

    print(f"Processing {len(data)} questions...")

    start_time = time.time()

    sem = asyncio.Semaphore(CONCURRENCY)

    async def bound_process(item, idx):
        async with sem:
            return await process_single_question(pipeline, item, idx, len(data))

    tasks = [bound_process(item, i) for i, item in enumerate(data)]
    results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    successful = [r for r in results if r.get("success")]
    avg_time = total_time / len(data) if data else 0
    total_passages = sum(r.get("num_passages", 0) for r in successful)
    avg_passages = total_passages / len(successful) if successful else 0

    print(f"\nDone in {total_time:.1f}s")
    print(f"Avg time per question: {avg_time:.1f}s")
    print(f"Avg passages used: {avg_passages:.1f}")

    output = {
        "config": {
            "pipeline_version": "v7",
            "dataset": args.dataset,
            "mode": "initial_sq + expanded_rerank_main_query + filters(relations.relation excluded, degree cutoff)",
            "concurrency": CONCURRENCY,
            "top_k": 3,
            "dense_weight": args.dense_weight,
            "bm25_weight": 1.0 - args.dense_weight,
            "max_docs_per_value": args.max_docs_per_value,
        },
        "summary": {
            "total_questions": len(data),
            "successful": len(successful),
            "errors": len(data) - len(successful),
            "total_time": total_time,
            "avg_time_per_question": avg_time,
            "total_passages": total_passages,
            "avg_passages_per_question": avg_passages,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(config["result_path"]), exist_ok=True)
    with open(config["result_path"], "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {config['result_path']}")


if __name__ == "__main__":
    asyncio.run(main())
