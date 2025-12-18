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
from Prompt.rag_qa_cot import fact_rag_qa_system, prompt_template_fact as prompt_template


COT_MAX_TOKENS = int(os.getenv("COT_MAX_TOKENS", "2048"))


def _extract_after_answer_marker(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    marker = "answer:"
    marker_idx = lower.rfind(marker)
    if marker_idx == -1:
        return text.strip()
    return text[marker_idx + len(marker):].strip()


def _build_rag_qa_cot_messages(prompt_user: str):
    messages = copy.deepcopy(prompt_template)
    messages[-1]["content"] = prompt_user
    return messages


def _build_prompt_user_rag_qa_template(
    passages,
    question: str,
    *,
    previous_context_text: str = "",
) -> str:
    """Build user content matching Prompt/rag_qa_cot.py one-shot format.

    Only real passages are formatted with `Wikipedia Title:`.
    Previous context is included as an explicit section.
    """
    parts = []
    # Make the instruction visible in FULL PROMPT (user request).
    parts.append("---Instruction---\n" + fact_rag_qa_system.strip())
    if previous_context_text and previous_context_text.strip():
        parts.append("---Previous Context---\n" + previous_context_text.strip())

    docs = ""
    for title, text in passages:
        docs += f"Wikipedia Title: {title}\n{text}\n"
    parts.append(docs.strip())

    return "\n\n".join([p for p in parts if p.strip()]) + f"\n\nQuestion: {question}\nThought: "


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


async def _call_cot(client: AsyncOpenAI, prompt_user: str):
    messages = _build_rag_qa_cot_messages(prompt_user)
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
    # SQ answering: do NOT include previous context (user request).
    prompt_user = _build_prompt_user_rag_qa_template(passages_for_template, actual_question)

    raw, messages = await _call_cot(client, prompt_user)
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

    all_passages = {}  # title -> {text, best_score}

    for sq in decomposition.subquestions:
        # SQ answering: always omit previous context.
        answer, results = await answer_subquestion(client, retriever, sq, decomposition, "", k=k)
        sq.answer = answer
        sq.retrieved_passages = results

        for res in results:
            title = res["title"]
            score = float(res.get('score', 0.0))
            if title not in all_passages or score > all_passages[title]['best_score']:
                all_passages[title] = {'text': res["text"], 'best_score': score}

    ranked = sorted(all_passages.items(), key=lambda kv: kv[1]["best_score"], reverse=True)
    top_passages = ranked[:5]

    passages_for_template = [(title, payload["text"]) for (title, payload) in top_passages]

    # Final answering: include ALL SQ Q/A as previous context.
    prev_context_text = _build_all_subqa_context(decomposition)
    prompt_user = _build_prompt_user_rag_qa_template(
        passages_for_template,
        question,
        previous_context_text=prev_context_text,
    )

    raw_final, messages = await _call_cot(client, prompt_user)
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
    parser.add_argument("--k", type=int, default=5, help="Number of passages per sub-question (default: 5)")
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
