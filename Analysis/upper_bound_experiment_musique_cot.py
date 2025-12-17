"""\
Upper Bound Experiment (MuSiQue) + CoT one-shot prompt
=====================================================

- Perfect Retrieval with ORIGINAL PASSAGES (supporting-fact titles -> full context passages)

This is a CoT-adapted version: it uses Prompt/rag_qa_cot.py `prompt_template`
(system + one-shot + assistant + user) and injects the actual input into
`${prompt_user}`.

Usage:
    python Analysis/upper_bound_experiment_musique_cot.py

Evaluation:
    python evaluate_mrqa_cot.py Results/upper_bound_musique_original_results_cot.json
"""

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Prompt.rag_qa_cot import prompt_template

load_dotenv()

# ==================== HYPERPARAMETERS ====================
CONCURRENCY = 50
MODEL = "openai/gpt-4o-mini"
MAX_TOKENS = 200
# =========================================================

DATASET = 'musique'
CONFIG = {
    'data_path': 'MuSiQue/musique_sample_200.json',
    'db_path': 'MuSiQue/metadata_v3.db',
    'result_original_cot': 'Results/upper_bound_musique_original_results_cot.json',
}


def _build_rag_qa_cot_messages(prompt_user: str):
    messages = copy.deepcopy(prompt_template)
    # last message must contain ${prompt_user}
    messages[-1]['content'] = prompt_user
    return messages


async def call_llm_cot(client: AsyncOpenAI, prompt_user: str) -> str:
    response = await client.chat.completions.create(
        model=MODEL,
        messages=_build_rag_qa_cot_messages(prompt_user),
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


async def process_question_original(
    client: AsyncOpenAI,
    sample: dict,
    semaphore: asyncio.Semaphore,
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

        prompt_user = f"---Information---\n{passages_text}\n\n---Query---\n{question}"
        predicted = await call_llm_cot(client, prompt_user)

        return {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "answer_aliases": sample.get("answer_aliases", []),
            "sf_titles": list(sf_info.keys()),
            "success": True,
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Upper Bound Experiments (MuSiQue) + CoT prompt')
    parser.add_argument('--max_questions', type=int, default=None, help='Limit number of questions (debug)')
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

    # Hop distribution (MuSiQue)
    hop_dist = {}
    for item in samples:
        hop = item['_id'].split('hop')[0]
        hop_dist[hop] = hop_dist.get(hop, 0) + 1
    print(f"Hop distribution: {hop_dist}")

    Path('Results').mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(CONCURRENCY)

    print("\n" + "=" * 60)
    print("Experiment: Perfect Retrieval with ORIGINAL PASSAGES (CoT)")
    print("=" * 60)
    start_time = time.time()
    tasks = [process_question_original(client, s, semaphore) for s in samples]
    results_original = await asyncio.gather(*tasks)
    time_original = time.time() - start_time

    with open(CONFIG['result_original_cot'], 'w', encoding='utf-8') as f:
        json.dump(
            {
                'experiment': f"{DATASET.upper()} Perfect Retrieval with Original Passages (CoT)",
                'dataset': DATASET,
                'model': MODEL,
                'prompt_mode': 'cot',
                'total': len(results_original),
                'time': time_original,
                'results': results_original,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Experiment 2 completed in {time_original:.1f}s")

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETED")
    print("=" * 60)
    print(f"Results saved to:")
    print(f"  - {CONFIG['result_original_cot']}")
    print("\nRun evaluation:")
    print(f"  python evaluate_mrqa_cot.py {CONFIG['result_original_cot']}")


if __name__ == '__main__':
    asyncio.run(main())
