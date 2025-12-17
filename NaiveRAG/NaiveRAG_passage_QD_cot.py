#!/usr/bin/env python3
"""\
Naive RAG - Passage Level (With Query Decomposition) + CoT one-shot
==================================================================
Uses Query Decomposition to retrieve passages for sub-questions, then synthesizes
the final answer using unique retrieved passages.

This is a CoT-adapted version: for sub-question answering and final answering,
it uses Prompt/rag_qa_cot.py `prompt_template` (one-shot) by injecting `${prompt_user}`.

Dataset-compatible (hotpotqa / musique).

Usage:
    python NaiveRAG/NaiveRAG_passage_QD_cot.py --dataset hotpotqa
    python NaiveRAG/NaiveRAG_passage_QD_cot.py --dataset musique

Evaluation:
    python evaluate_mrqa_cot.py Results/NaiveRAG/NaiveRAG_passage_QD_cot_hotpotqa.json
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
from query_decomposition import decompose_query, substitute_answers
from llm_logger import init_logger, log_llm_call
from Prompt.rag_qa_cot import prompt_template


def _build_rag_qa_cot_messages(prompt_user: str):
    messages = copy.deepcopy(prompt_template)
    messages[-1]["content"] = prompt_user
    return messages


async def _call_cot(client: AsyncOpenAI, prompt_user: str) -> str:
    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=_build_rag_qa_cot_messages(prompt_user),
        temperature=0.0,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


async def answer_subquestion(client, retriever, sq, decomposition, previous_context):
    actual_question = substitute_answers(sq.question, decomposition.subquestions)

    results = await retriever.search(actual_question, k=3)

    passage_text = ""
    for i, res in enumerate(results, 1):
        passage_text += f"[{i}] {res['title']}\n{res['text']}\n\n"

    info_blocks = []
    if previous_context.strip():
        info_blocks.append("---Previous Context---\n" + previous_context.strip())
    info_blocks.append("---Passages---\n" + passage_text.strip())

    prompt_user = "---Information---\n" + "\n\n".join(info_blocks) + f"\n\n---Query---\n{actual_question}"

    answer = await _call_cot(client, prompt_user)

    log_llm_call(
        call_type="SubQuestion Answering (NaiveRAG CoT)",
        input_text=prompt_user,
        output_text=answer,
        context={
            "subquestion": actual_question,
            "num_passages": len(results),
            "passages": [r["title"] for r in results],
        },
    )

    return answer, results


async def process_question(client, retriever, item):
    question = item["question"]

    decomposition_result = await decompose_query(client, question)
    if not decomposition_result or not decomposition_result.get("success"):
        return {"id": item.get("_id"), "question": question, "error": "Decomposition failed"}

    decomposition = decomposition_result["decomposition"]

    all_passages = {}

    for sq in decomposition.subquestions:
        prev_context = ""
        if sq.depends_on:
            for dep_id in sq.depends_on:
                dep_sq = decomposition.get_subquestion(dep_id)
                if dep_sq and dep_sq.answer:
                    prev_context += f"Q: {dep_sq.question}\nA: {dep_sq.answer}\n\n"

        answer, results = await answer_subquestion(client, retriever, sq, decomposition, prev_context)
        sq.answer = answer
        sq.retrieved_passages = results

        for res in results:
            all_passages[res["title"]] = res["text"]

    unique_passages_text = ""
    for i, (title, text) in enumerate(all_passages.items(), 1):
        unique_passages_text += f"[{i}] {title}\n{text}\n\n"

    chain_text = ""
    for sq in decomposition.subquestions:
        chain_text += f"{sq.id}: {sq.question}\n{sq.answer}\n\n"

    final_info = "---Subquestion Chain---\n" + chain_text.strip() + "\n\n---Passages---\n" + unique_passages_text.strip()
    prompt_user = f"---Information---\n{final_info}\n\n---Query---\n{question}"

    final_answer = await _call_cot(client, prompt_user)

    log_llm_call(
        call_type="Final Answer Synthesis (NaiveRAG CoT)",
        input_text=prompt_user,
        output_text=final_answer,
        context={
            "main_query": question,
            "num_unique_passages": len(all_passages),
        },
    )

    return {
        "id": item.get("_id"),
        "question": question,
        "gold_answer": item["answer"],
        "predicted_answer": final_answer,
        "answer_aliases": item.get("answer_aliases", []),
        "decomposition": [sq.to_dict() for sq in decomposition.subquestions],
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hotpotqa", choices=["hotpotqa", "musique"])
    args = parser.parse_args()

    load_dotenv()
    init_logger()

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
    print(f"Processing {len(data)} questions with QD (CoT)...")

    start_time = time.time()

    batch_size = 50
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        tasks = [process_question(chat_client, retriever, item) for item in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        print(f"Processed {min(i + batch_size, len(data))}/{len(data)}")

    total_time = time.time() - start_time

    output_file = f"Results/NaiveRAG/NaiveRAG_passage_QD_cot_{args.dataset}.json"
    os.makedirs("Results/NaiveRAG", exist_ok=True)

    output = {
        "config": {
            "pipeline": "NaiveRAG_QD_CoT",
            "dataset": args.dataset,
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
