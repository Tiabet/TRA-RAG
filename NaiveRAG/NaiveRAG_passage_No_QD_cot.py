#!/usr/bin/env python3
"""\
Naive RAG - Passage Level (No Query Decomposition) + CoT one-shot
================================================================
Retrieves passages directly using vector embeddings and answers using
Prompt/answer_prompt.py.

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
from Prompt.answer_prompt import NAIVE_RAG_FINAL_ANSWER_PROMPT
from llm_logger import init_logger, finalize_log, log_llm_call


COT_MAX_TOKENS = int(os.getenv("COT_MAX_TOKENS", "256"))


def _extract_answer(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.lower().startswith("answer:"):
        return t.split(":", 1)[1].strip()
    return t


async def process_question(client, retriever, item, k):
    question = item["question"]

    # Retrieve
    results = await retriever.search(question, k=k)

    # Format passages
    passage_text = ""
    for i, res in enumerate(results, 1):
        passage_text += f"[{i}] {res['title']}\n{res['text']}\n\n"

    prompt_user = NAIVE_RAG_FINAL_ANSWER_PROMPT.replace("{{passages}}", passage_text.strip())
    prompt_user = prompt_user.replace("{{question}}", question)
    call_type = "Answer Generation (NaiveRAG No_QD)"

    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise question answering system."},
            {"role": "user", "content": prompt_user},
        ],
        temperature=0.0,
        max_tokens=COT_MAX_TOKENS,
    )

    raw = response.choices[0].message.content.strip()
    answer = _extract_answer(raw)

    log_llm_call(
        call_type=call_type,
        input_text=prompt_user,
        output_text=raw,
        context={
            "question": question,
            "k": k,
            "num_passages": len(results),
            "passages": [r['title'] for r in results],
        },
    )

    qid = item.get("_id") or item.get("id")

    final_retrieved_passages = [
        {
            "doc_id": r.get("doc_id"),
            "title": r.get("title"),
            "score": r.get("score"),
        }
        for r in results
    ]

    return {
        "id": qid,
        "question": question,
        "gold_answer": item["answer"],
        "predicted_answer": answer,
        "predicted_answer_raw": raw,
        "answer_aliases": item.get("answer_aliases", []),
        "final_retrieved_passages": final_retrieved_passages,
        "retrieved_passages": [{"doc_id": r.get("doc_id"), "title": r.get("title")} for r in results],
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Number of passages to retrieve (fixed at 5 by default)")
    parser.add_argument("--dataset", type=str, default="hotpotqa", choices=["hotpotqa", "musique"])
    args = parser.parse_args()

    load_dotenv()
    init_logger()

    if args.dataset == "hotpotqa":
        data_path = "HotpotQA/hotpotqa_sample_200_corpus_idx.json"
        cache_path = "HotpotQA/passage_embeddings_sample_200.npz"
    else:
        data_path = "MuSiQue/musique_sample_200_corpus_idx.json"
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
    print(f"Processing {len(data)} questions with K={args.k} (rag_qa_template)...")

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
            "prompt_variant": "answer_prompt",
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
    log_path = finalize_log()
    print(f"[LOG] {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
