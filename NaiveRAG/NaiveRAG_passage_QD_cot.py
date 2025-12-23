#!/usr/bin/env python3
"""\
Naive RAG - Passage Level (With Query Decomposition) + CoT one-shot
==================================================================
Uses Query Decomposition to retrieve passages for sub-questions, then synthesizes
the final answer using unique retrieved passages.

This keeps the CoT runner structure (Answer-only extraction + raw capture),
but uses Prompt/answer_prompt.py.

Dataset-compatible (hotpotqa / musique).

Usage:
    python NaiveRAG/NaiveRAG_passage_QD_cot.py --dataset hotpotqa --k 5
    python NaiveRAG/NaiveRAG_passage_QD_cot.py --dataset musique --k 5

Evaluation:
    python evaluate_mrqa_cot.py Results/NaiveRAG/NaiveRAG_passage_QD_cot_hotpotqa_k5.json
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
from llm_logger import init_logger, finalize_log, log_llm_call
from Prompt.answer_prompt import (
    DETAILED_SUBQUESTION_ANSWERING_PROMPT,
    FINAL_ANSWER_SYNTHESIS_PROMPT,
)


COT_MAX_TOKENS = int(os.getenv("COT_MAX_TOKENS", "256"))


def _extract_after_answer_marker(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    marker = "answer:"
    marker_idx = lower.rfind(marker)
    if marker_idx == -1:
        return text.strip()
    return text[marker_idx + len(marker):].strip()


def _build_subquestion_prompt(
    passages,
    question: str,
    *,
    main_query: str,
    previous_context_text: str = "",
) -> str:
    passage_text = ""
    for i, (title, text) in enumerate(passages, 1):
        passage_text += f"[{i}] {title}\n{text}\n\n"

    prompt = DETAILED_SUBQUESTION_ANSWERING_PROMPT
    prompt = prompt.replace("{{main_query}}", main_query)
    prompt = prompt.replace("{{previous_context}}", previous_context_text.strip() if previous_context_text else "None")
    prompt = prompt.replace("{{passages}}", passage_text.strip() if passage_text else "No passages retrieved.")
    prompt = prompt.replace("{{subquestion}}", question)
    return prompt


def _build_final_prompt(
    *,
    main_question: str,
    subquestion_chain: str,
    passages,
) -> str:
    passage_text = ""
    for i, (title, text) in enumerate(passages, 1):
        passage_text += f"[{i}] {title}\n{text}\n\n"
    prompt = FINAL_ANSWER_SYNTHESIS_PROMPT
    prompt = prompt.replace("{{main_question}}", main_question)
    prompt = prompt.replace("{{subquestion_chain}}", subquestion_chain.strip() if subquestion_chain else "")
    prompt = prompt.replace("{{passages}}", passage_text.strip() if passage_text else "No passages retrieved.")
    return prompt


def _build_all_subqa_context(decomposition) -> str:
    context_parts = []
    for sq in decomposition.subquestions:
        if getattr(sq, 'answer', None):
            context_parts.append(f"{sq.id}: {sq.question}")
            context_parts.append(f"Answer: {sq.answer}")
            context_parts.append("")
    if context_parts:
        return "Previous Sub-Questions:\n" + "\n".join(context_parts)
    return ""


async def _call_llm(client: AsyncOpenAI, prompt_user: str):
    messages = [
        {"role": "system", "content": "You are a precise question answering system."},
        {"role": "user", "content": prompt_user},
    ]
    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=COT_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip(), messages


async def answer_subquestion(client, retriever, sq, decomposition, previous_context, k: int):
    actual_question = substitute_answers(sq.question, decomposition.subquestions)

    results = await retriever.search(actual_question, k=k)

    passages_for_template = [(r["title"], r["text"]) for r in results]
    prompt_user = _build_subquestion_prompt(
        passages_for_template,
        actual_question,
        main_query=decomposition.main_query,
        previous_context_text=previous_context or "",
    )

    raw, messages = await _call_llm(client, prompt_user)
    answer = _extract_after_answer_marker(raw)

    log_llm_call(
        call_type="SubQuestion Answering (NaiveRAG CoT)",
        input_text=prompt_user,
        output_text=raw,
        context={
            "subquestion": actual_question,
            "num_passages": len(results),
            "passages": [r["title"] for r in results],
            "system_message": messages[0].get("content", "") if messages else "",
            "one_shot_user": messages[1].get("content", "") if isinstance(messages, list) and len(messages) > 1 else "",
            "one_shot_assistant": messages[2].get("content", "") if isinstance(messages, list) and len(messages) > 2 else "",
        },
    )

    return answer, results


async def process_question(client, retriever, item, k: int):
    question = item["question"]

    decomposition_result = await decompose_query(client, question)
    if not decomposition_result or not decomposition_result.get("success"):
        return {"id": item.get("_id"), "question": question, "error": "Decomposition failed"}

    decomposition = decomposition_result["decomposition"]

    all_passages = {}  # doc_id -> {title, text, best_score}

    for sq in decomposition.subquestions:
        # Build previous context from dependencies (same behavior as non-CoT NaiveRAG_passage_QD.py)
        prev_context = ""
        if getattr(sq, 'depends_on', None):
            for dep_id in sq.depends_on:
                dep_sq = decomposition.get_subquestion(dep_id)
                if dep_sq and getattr(dep_sq, 'answer', None):
                    prev_context += f"Q: {dep_sq.question}\nA: {dep_sq.answer}\n\n"

        answer, results = await answer_subquestion(client, retriever, sq, decomposition, prev_context, k=k)
        sq.answer = answer
        sq.retrieved_passages = results

        for res in results:
            doc_id = str(res.get('doc_id')) if res.get('doc_id') is not None else ''
            if not doc_id:
                continue
            title = res.get("title")
            score = float(res.get('score', 0.0))
            if doc_id not in all_passages or score > all_passages[doc_id]['best_score']:
                all_passages[doc_id] = {'title': title, 'text': res["text"], 'best_score': score}

    ranked = sorted(all_passages.items(), key=lambda kv: kv[1]["best_score"], reverse=True)
    top_passages = ranked[:5]

    passages_for_template = [(payload.get('title') or '', payload["text"]) for (_doc_id, payload) in top_passages]

    final_retrieved_passages = [
        {
            'doc_id': str(doc_id),
            'title': payload.get('title') or '',
            'score': float(payload.get('best_score', 0.0)),
        }
        for (doc_id, payload) in top_passages
    ]

    # Final answering: include ALL SQ Q/A as chain (fits FINAL_ANSWER_SYNTHESIS_PROMPT).
    chain_text = ""
    for sq in decomposition.subquestions:
        chain_text += f"{sq.id}: {sq.question}\nAnswer: {sq.answer}\n\n"

    prompt_user = _build_final_prompt(
        main_question=question,
        subquestion_chain=chain_text,
        passages=passages_for_template,
    )

    raw_final, messages = await _call_llm(client, prompt_user)
    final_answer = _extract_after_answer_marker(raw_final)

    log_llm_call(
        call_type="Final Answer Synthesis (NaiveRAG CoT)",
        input_text=prompt_user,
        output_text=raw_final,
        context={
            "main_query": question,
            "num_unique_passages": len(all_passages),
            "system_message": messages[0].get("content", "") if messages else "",
            "one_shot_user": messages[1].get("content", "") if isinstance(messages, list) and len(messages) > 1 else "",
            "one_shot_assistant": messages[2].get("content", "") if isinstance(messages, list) and len(messages) > 2 else "",
        },
    )

    return {
        "id": item.get("_id") or item.get('id'),
        "question": question,
        "gold_answer": item["answer"],
        "predicted_answer": final_answer,
        "predicted_answer_raw": raw_final,
        "answer_aliases": item.get("answer_aliases", []),
        "final_retrieved_passages": final_retrieved_passages,
        "decomposition": [sq.to_dict() for sq in decomposition.subquestions],
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hotpotqa", choices=["hotpotqa", "musique"])
    parser.add_argument("--k", type=int, default=5, help="Number of passages per sub-question (default: 5)")
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
    print(f"Processing {len(data)} questions with QD (CoT, k={args.k})...")

    start_time = time.time()

    batch_size = 100
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        tasks = [process_question(chat_client, retriever, item, k=args.k) for item in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        print(f"Processed {min(i + batch_size, len(data))}/{len(data)}")

    total_time = time.time() - start_time

    output_file = f"Results/NaiveRAG/NaiveRAG_passage_QD_cot_{args.dataset}_k{args.k}.json"
    os.makedirs("Results/NaiveRAG", exist_ok=True)

    output = {
        "config": {
            "pipeline": "NaiveRAG_QD_CoT",
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
    log_path = finalize_log()
    print(f"[LOG] {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
