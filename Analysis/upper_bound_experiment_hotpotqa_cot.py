"""\
Upper Bound Experiment (HotpotQA) + CoT one-shot prompt
======================================================

- Perfect Retrieval with ORIGINAL PASSAGES (supporting-fact titles -> full context passages)

This is a CoT-adapted version: it uses Prompt/rag_qa_cot.py `prompt_template`
(system + one-shot + assistant + user) and injects the actual input into
`${prompt_user}`.

Usage:
    python Analysis/upper_bound_experiment_hotpotqa_cot.py
    python Analysis/upper_bound_experiment_hotpotqa_cot.py --max_questions 20

Evaluation:
    python evaluate_mrqa_cot.py Results/upper_bound_hotpotqa_original_results_cot_template.json
"""

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from llm_logger import init_logger, finalize_log, log_llm_call

from Prompt.rag_qa_cot import prompt_template as RAG_QA_TEMPLATE

load_dotenv()

# ==================== HYPERPARAMETERS ====================
CONCURRENCY = 100
MODEL = "openai/gpt-4o-mini"
MAX_TOKENS = 200
# =========================================================

DATASET = 'hotpotqa'
CONFIG = {
    'data_path': 'HotpotQA/hotpotqa_sample_200.json',
    'result_cot_template': 'Results/upper_bound_hotpotqa_original_results_cot_template.json',
}

VARIANT_RAG_QA_TEMPLATE = 'rag_qa_template'

# Backward-compatible alias
_ALIASES = {
    'cot_template': VARIANT_RAG_QA_TEMPLATE,
}


def _build_messages(prompt_user: str, template: List[dict]):
    messages = copy.deepcopy(template)
    messages[-1]['content'] = prompt_user
    return messages


def _format_chat_messages_for_log(messages: List[dict]) -> str:
    parts: List[str] = []
    for i, m in enumerate(messages, 1):
        role = m.get('role', 'unknown')
        content = m.get('content', '')
        parts.append(f"--- message {i} ({role}) ---\n{content}")
    return "\n\n".join(parts)


def _build_prompt_user_cot_template(passages: List[Tuple[str, str]], question: str) -> str:
    docs = ''
    for title, text in passages:
        docs += f"Wikipedia Title: {title}\n{text}\n"
    return f"{docs}\n\nQuestion: {question}\nThought: "


async def call_llm_cot(client: AsyncOpenAI, messages: List[dict]) -> str:
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


async def process_question_original(
    client: AsyncOpenAI,
    sample: dict,
    semaphore: asyncio.Semaphore,
    variant: str,
) -> dict:
    async with semaphore:
        question = sample["question"]
        gold_answer = sample["answer"]
        qid = sample.get("_id", "")

        sf_info = {}
        for sf in sample["supporting_facts"]:
            title, sent_idx = sf[0], sf[1]
            if title not in sf_info:
                sf_info[title] = set()
            sf_info[title].add(sent_idx)

        passages_text = ""
        context_dict = {ctx[0]: ctx[1] for ctx in sample["context"]}

        for i, (title, _sent_indices) in enumerate(sf_info.items(), 1):
            if title in context_dict:
                sentences = context_dict[title]
                full_passage = "".join(sentences) if isinstance(sentences, list) else sentences
                passages_text += f"[{i}] {title}\n{full_passage}\n\n"

        if not passages_text:
            return {
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": "Insufficient information.",
                "answer_aliases": sample.get("answer_aliases", []),
                "sf_titles": list(sf_info.keys()),
                "success": False,
            }

        passages_for_template: List[Tuple[str, str]] = []
        for title, _sent_indices in sf_info.items():
            if title not in context_dict:
                continue
            sentences = context_dict[title]
            full_passage = "".join(sentences) if isinstance(sentences, list) else str(sentences)
            full_passage = full_passage.strip()
            if full_passage:
                passages_for_template.append((title, full_passage))

        if variant != VARIANT_RAG_QA_TEMPLATE:
            raise ValueError(f"Unknown variant: {variant}")

        prompt_user = _build_prompt_user_cot_template(passages_for_template, question)
        messages = _build_messages(prompt_user, RAG_QA_TEMPLATE)
        call_type = 'Upper Bound Original Passages (rag_qa_template)'

        predicted = await call_llm_cot(client, messages)

        full_prompt_for_log = _format_chat_messages_for_log(messages)

        log_llm_call(
            call_type=call_type,
            input_text=full_prompt_for_log,
            output_text=predicted,
            context={
                "dataset": DATASET,
                "id": qid,
                "num_sf_titles": len(sf_info),
                "sf_titles": list(sf_info.keys()),
                "prompt_variant": variant,
            },
        )

        return {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "answer_aliases": sample.get("answer_aliases", []),
            "sf_titles": list(sf_info.keys()),
            "prompt_variant": variant,
            "success": True,
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Upper Bound Experiments (HotpotQA) + CoT prompt')
    parser.add_argument('--max_questions', type=int, default=None, help='Limit number of questions (debug)')
    parser.add_argument(
        '--variant',
        type=str,
        default=VARIANT_RAG_QA_TEMPLATE,
        choices=[VARIANT_RAG_QA_TEMPLATE, 'cot_template'],
        help='Which CoT prompting variant to run',
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL'),
    )

    print("=" * 80)
    print(f"Upper Bound Experiments (CoT) - {DATASET.upper()}")
    print("=" * 80)

    with open(CONFIG['data_path'], 'r', encoding='utf-8') as f:
        samples = json.load(f)

    if args.max_questions is not None:
        samples = samples[: max(0, args.max_questions)]

    print(f"Loaded {len(samples)} samples from {CONFIG['data_path']}")

    Path('Results').mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(CONCURRENCY)

    variant = _ALIASES.get(args.variant, args.variant)

    print("\n" + "=" * 60)
    print(f"Experiment: Perfect Retrieval with ORIGINAL PASSAGES ({variant})")
    print("=" * 60)

    start_time = time.time()
    tasks = [process_question_original(client, s, semaphore, variant=variant) for s in samples]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    out_path = CONFIG['result_cot_template']
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'experiment': f"{DATASET.upper()} Perfect Retrieval with Original Passages ({variant})",
                'dataset': DATASET,
                'model': MODEL,
                'prompt_mode': 'cot',
                'prompt_variant': variant,
                'total': len(results),
                'time': elapsed,
                'results': results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved: {out_path} ({elapsed:.1f}s)")

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETED")
    print("=" * 60)
    print(f"Results saved to:")
    print(f"  - {CONFIG['result_cot_template']}")

    print("\nRun evaluation:")
    print(f"  python evaluate_mrqa_cot.py {CONFIG['result_cot_template']}")


if __name__ == '__main__':
    init_logger()
    try:
        asyncio.run(main())
    finally:
        finalize_log()
