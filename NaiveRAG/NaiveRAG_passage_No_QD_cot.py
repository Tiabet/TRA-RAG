#!/usr/bin/env python3
"""\
Naive RAG - Passage Level (No Query Decomposition) + CoT one-shot
================================================================
Retrieves passages directly using vector embeddings and answers using
Prompt/rag_qa_cot.py `prompt_template` (one-shot) by injecting `${prompt_user}`.

Dataset-compatible (hotpotqa / musique).

Usage:
    python NaiveRAG/NaiveRAG_passage_No_QD_cot.py --dataset hotpotqa --k 5
    python NaiveRAG/NaiveRAG_passage_No_QD_cot.py --dataset musique --k 5

Evaluation:
    python evaluate_mrqa_cot.py Results/NaiveRAG/NaiveRAG_passage_No_QD_cot_hotpotqa_k5.json
"""

import argparse
import asyncio
import copy
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NaiveRAG.naive_passage_retriever import NaivePassageRetriever
from Prompt.rag_qa_cot import prompt_template


def _build_rag_qa_cot_messages(prompt_user: str):
    messages = copy.deepcopy(prompt_template)
    messages[-1]["content"] = prompt_user
    return messages


async def process_question(client, retriever, item, k):
    question = item["question"]

    # Retrieve
    results = await retriever.search(question, k=k)

    # Format passages
    passage_text = ""
    for i, res in enumerate(results, 1):
        passage_text += f"[{i}] {res['title']}\n{res['text']}\n\n"

    prompt_user = f"---Information---\n{passage_text}\n\n---Query---\n{question}"

    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=_build_rag_qa_cot_messages(prompt_user),
        temperature=0.0,
        max_tokens=200,
    )

    answer = response.choices[0].message.content.strip()

    return {
        "id": item.get("_id"),
        "question": question,
        "gold_answer": item["answer"],
        "predicted_answer": answer,
        "answer_aliases": item.get("answer_aliases", []),
        "retrieved_passages": [{"title": r["title"]} for r in results],
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="Number of passages to retrieve")
    parser.add_argument("--dataset", type=str, default="hotpotqa", choices=["hotpotqa", "musique"])
    args = parser.parse_args()

    load_dotenv()

    if args.dataset == "hotpotqa":
        data_path = "HotpotQA/hotpotqa_sample_200.json"
        cache_path = "HotpotQA/passage_embeddings_sample_200.npz"
    else:
        data_path = "MuSiQue/musique_sample_200.json"
        cache_path = "MuSiQue/passage_embeddings_sample_200.npz"

    chat_client = AsyncOpenAI(
        api_key=os.getenv("ALICE_OPENAI_KEY"),
        base_url=os.getenv("ALICE_CHAT_URL"),
    )

    embed_client = AsyncOpenAI(
        api_key=os.getenv("ALICE_OPENAI_KEY"),
        base_url=os.getenv("ALICE_EMBED_URL"),
    )

    retriever = NaivePassageRetriever(embed_client, data_path, cache_path)
    await retriever.initialize()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    print(f"Processing {len(data)} questions with K={args.k} (CoT)...")

    start_time = time.time()

    batch_size = 50
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        tasks = [process_question(chat_client, retriever, item, args.k) for item in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        print(f"Processed {min(i + batch_size, len(data))}/{len(data)}")

    total_time = time.time() - start_time

    output_file = f"Results/NaiveRAG/NaiveRAG_passage_No_QD_cot_{args.dataset}_k{args.k}.json"
    os.makedirs("Results/NaiveRAG", exist_ok=True)

    output = {
        "config": {
            "pipeline": "NaiveRAG_No_QD_CoT",
            "dataset": args.dataset,
            "k": args.k,
            "model": "gpt-4o-mini",
        },
        "summary": {
            "total_questions": len(data),
            "total_time": total_time,
            "avg_time_per_question": total_time / len(data) if data else 0,
        },
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
