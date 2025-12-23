"""\
Upper Bound Experiment (MuSiQue)
=====================================================

- Perfect Retrieval with ORIGINAL PASSAGES (supporting-fact titles -> full context passages)

Uses Prompt/answer_prompt.py.

Usage:
    python Analysis/upper_bound_experiment_musique_cot.py
    python Analysis/upper_bound_experiment_musique_cot.py --max_questions 20

Evaluation:
    python evaluate_mrqa.py Results/upper_bound_musique_original_results_cot_template.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from llm_logger import init_logger, finalize_log, log_llm_call

from Prompt.answer_prompt import NAIVE_RAG_FINAL_ANSWER_PROMPT

load_dotenv()

# ==================== HYPERPARAMETERS ====================
CONCURRENCY = 50
MODEL = "openai/gpt-4o-mini"
MAX_TOKENS = 200
# =========================================================

DATASET = 'musique'
CONFIG = {
    'data_path': 'MuSiQue/musique_sample_200.json',
    'result_cot_template': 'Results/upper_bound_musique_original_results_cot_template.json',
}

PROMPT_VARIANT = 'answer_prompt'

def _extract_answer(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.lower().startswith("answer:"):
        return t.split(":", 1)[1].strip()
    return t


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

        # NOTE (MuSiQue): the same title may appear multiple times in `context`.
        # A plain dict {title: sentences} will overwrite earlier entries and can drop the true gold passage.
        context_by_title = defaultdict(list)  # title -> list[list[str]]
        for title, sentences in sample.get("context", []):
            context_by_title[title].append(sentences)

        passages_text = ""
        injected_titles = []
        ambiguous_titles = []
        passage_counter = 0
        for title, _sent_indices in sf_info.items():
            occurrences = context_by_title.get(title, [])
            if not occurrences:
                continue
            injected_titles.append(title)
            if len(occurrences) > 1:
                ambiguous_titles.append(title)

            # Dedup identical passage texts for the same title
            seen_texts = set()
            for occ_idx, sents in enumerate(occurrences, 1):
                full_passage = "".join(sents) if isinstance(sents, list) else str(sents)
                full_passage = full_passage.strip()
                if not full_passage:
                    continue
                if full_passage in seen_texts:
                    continue
                seen_texts.add(full_passage)

                passage_counter += 1
                suffix = f" (occurrence {occ_idx})" if len(occurrences) > 1 else ""
                passages_text += f"[{passage_counter}] {title}{suffix}\n{full_passage}\n\n"

        if not passages_text:
            return {
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": "Insufficient information.",
                "answer_aliases": sample.get("answer_aliases", []),
                "sf_titles": list(sf_info.keys()),
                "sf_titles_injected": [],
                "sf_titles_ambiguous_in_context": [],
                "num_injected_passages": 0,
                "success": False,
            }

        # Build prompt_user per variant
        passages_for_template: List[Tuple[str, str]] = []
        prompt_user = NAIVE_RAG_FINAL_ANSWER_PROMPT.replace("{{passages}}", passages_text.strip())
        prompt_user = prompt_user.replace("{{question}}", question)
        messages = [
            {"role": "system", "content": "You are a precise question answering system."},
            {"role": "user", "content": prompt_user},
        ]
        call_type = 'Upper Bound Original Passages (answer_prompt)'

        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=MAX_TOKENS,
        )
        predicted_raw = response.choices[0].message.content.strip()
        predicted = _extract_answer(predicted_raw)

        full_prompt_for_log = prompt_user

        log_llm_call(
            call_type=call_type,
            input_text=full_prompt_for_log,
            output_text=predicted,
            context={
                "dataset": DATASET,
                "id": qid,
                "num_sf_titles": len(sf_info),
                "sf_titles": list(sf_info.keys()),
                "sf_titles_ambiguous_in_context": ambiguous_titles,
                "num_injected_passages": passage_counter,
                "prompt_variant": PROMPT_VARIANT,
            },
        )

        return {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "answer_aliases": sample.get("answer_aliases", []),
            "sf_titles": list(sf_info.keys()),
            "sf_titles_injected": injected_titles,
            "sf_titles_ambiguous_in_context": ambiguous_titles,
            "num_injected_passages": passage_counter,
            "prompt_variant": PROMPT_VARIANT,
            "success": True,
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Upper Bound Experiments (MuSiQue)')
    parser.add_argument('--max_questions', type=int, default=None, help='Limit number of questions (debug)')
    return parser.parse_args()


async def main():
    args = parse_args()

    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL'),
    )

    print("=" * 80)
    print(f"Upper Bound Experiments - {DATASET.upper()}")
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
    print(f"Experiment: Perfect Retrieval with ORIGINAL PASSAGES ({PROMPT_VARIANT})")
    print("=" * 60)

    start_time = time.time()
    tasks = [process_question_original(client, s, semaphore, variant=PROMPT_VARIANT) for s in samples]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    out_path = CONFIG['result_cot_template']
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'experiment': f"{DATASET.upper()} Perfect Retrieval with Original Passages ({PROMPT_VARIANT})",
                'dataset': DATASET,
                'model': MODEL,
                'prompt_mode': 'standard',
                'prompt_variant': PROMPT_VARIANT,
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
    print(f"  python evaluate_mrqa.py {CONFIG['result_cot_template']}")


if __name__ == '__main__':
    init_logger()
    try:
        asyncio.run(main())
    finally:
        finalize_log()
