"""
Upper Bound Experiments
========================
Perfect Retrieval with Original Passages (Gold supporting passages)

Usage:
    python upper_bound_experiment.py                     # Default: HotpotQA
    python upper_bound_experiment.py --dataset hotpotqa  # HotpotQA
    python upper_bound_experiment.py --dataset musique   # MuSiQue
    python upper_bound_experiment.py --dataset 2wiki     # 2WikiMultihopQA

Optional:
    python upper_bound_experiment.py --dataset hotpotqa --prompt naive
        - Uses the repo's NAIVE_RAG_ANSWER_PROMPT
    python upper_bound_experiment.py --dataset hotpotqa --prompt final
        - Uses the repo's FINAL_ANSWER_SYNTHESIS_PROMPT (same as current pipeline)
    python upper_bound_experiment.py --dataset hotpotqa --prompt upper
        - Uses the local UPPER_BOUND_PROMPT (legacy)
"""

import json
import asyncio
import os
import sys
import time
import argparse
import hashlib
import random
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

from llm_provider import create_async_chat_client, detect_provider

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Prompt.answer_prompt import FINAL_ANSWER_SYNTHESIS_PROMPT, NAIVE_RAG_ANSWER_PROMPT

load_dotenv()

# ==================== HYPERPARAMETERS ====================
CONCURRENCY = 50
_PROVIDER_CFG = detect_provider()
MODEL = _PROVIDER_CFG.chat_model
# =========================================================

# Dataset configurations
DATASET_CONFIGS = {
    'hotpotqa': {
        'data_path': 'HotpotQA/hotpotqa.json',
        'result_original': 'Results/hotpot_upper_legacy.json',
        'result_original_naive': 'Results/hotpot_upper.json',
        'result_original_final': 'Results/hotpot_upper_final.json',
    },
    'musique': {
        'data_path': 'MuSiQue/musique.json',
        'result_original': 'Results/musique_upper_legacy.json',
        'result_original_naive': 'Results/musique_upper.json',
        'result_original_final': 'Results/musique_upper_final.json',
    },
    # NOTE: This expects a QA-style 2Wiki JSON where each item includes:
    # - question, answer
    # - paragraphs: list[{title, paragraph_text, is_supporting, ...}]
    # If your local file is corpus-only (no question/answer), pass a proper QA file via --data_path.
    '2wiki': {
        'data_path': '2WikiMultihopQA/2wikimultihopqa.json',
        'result_original': 'Results/2wiki_upper_legacy.json',
        'result_original_naive': 'Results/2wiki_upper.json',
        'result_original_final': 'Results/2wiki_upper_final.json',
    }
}

# Initialize client (OpenAI: no base_url; ALICE: uses ALICE_CHAT_URL)
client = create_async_chat_client(_PROVIDER_CFG)

# ==================== PROMPTS ====================
# IMPORTANT: Prompt selection is centralized in `build_prompt()`.
# - prompt=naive  -> NAIVE_RAG_ANSWER_PROMPT (repo)
# - prompt=final  -> FINAL_ANSWER_SYNTHESIS_PROMPT (repo, same as pipeline)
# - prompt=upper  -> UPPER_BOUND_PROMPT (legacy local template)

UPPER_BOUND_PROMPT = """---Role---
You are a multi-hop retrieval-augmented assistant.

---Goal---
Read the provided Information and generate the correct answer to the Question.
Use ONLY the given Information to derive your answer.

---Critical Instructions---
1. Read ALL provided passages carefully
2. Extract relevant facts from the passages
3. Combine information from multiple passages if needed
4. Perform simple reasoning if required (arithmetic, temporal logic, comparisons)

---Target response length and format---
- One-word or minimal-phrase answer (max 5 words).

---Output Constraint (Strict)---
- Output ONLY the answer text.
- Do NOT include emojis, bullet points, or any extra commentary.

---Response Rules---
✓ Use ONLY the information provided in the passages
✓ Check ALL passages thoroughly
✓ You CAN perform simple reasoning on passage information
✓ Answer must be short and concise
✓ Answer language must match the Question language
✗ Do NOT use external knowledge not present in passages
✗ Do NOT hallucinate or invent facts
✗ ONLY respond "Insufficient information." if passages truly lack the needed information

---Information---
{passages}

---Question---
{question}

---Answer---
Provide only the answer (max 5 words).
ONLY respond "Insufficient information." if the passages truly lack the needed information.
"""


def build_prompt(prompt_mode: str, question: str, passages_text: str) -> str:
    """Return the final prompt string for the given mode.

    Modes:
    - naive: NAIVE_RAG_ANSWER_PROMPT (repo)
    - final: FINAL_ANSWER_SYNTHESIS_PROMPT (repo)
    - upper: UPPER_BOUND_PROMPT (legacy)
    """
    mode = (prompt_mode or "naive").lower().strip()

    if mode == 'naive':
        prompt = NAIVE_RAG_ANSWER_PROMPT.replace("{{question}}", question)
        prompt = prompt.replace("{{passages}}", passages_text if passages_text else "No passages.")
        return prompt

    if mode == 'final':
        # Match the pipeline's final-answer prompt shape.
        subquestion_chain = "N/A (Gold supporting facts provided directly)"
        prompt = FINAL_ANSWER_SYNTHESIS_PROMPT.replace("{{main_question}}", question)
        prompt = prompt.replace("{{subquestion_chain}}", subquestion_chain)
        prompt = prompt.replace("{{passages}}", passages_text if passages_text else "No passages.")
        return prompt

    # mode == 'upper' (or any unknown fallback): use local legacy upper-bound template.
    return UPPER_BOUND_PROMPT.format(passages=passages_text, question=question)


def _stable_seed(value: str) -> int:
    """Stable per-question seed for reproducible 'random' noise passage selection."""
    digest = hashlib.md5(value.encode('utf-8')).hexdigest()
    return int(digest[:8], 16)


def _format_passages(passages: list[dict]) -> str:
    """Format passages as the prompt expects, preserving order."""
    chunks: list[str] = []
    for idx, p in enumerate(passages, 1):
        title = (p.get('title') or '').strip()
        text = (p.get('text') or '').strip()
        if not text:
            continue
        header = f"[{idx}] {title}" if title else f"[{idx}]"
        chunks.append(f"{header}\n{text}")
    return "\n\n".join(chunks)


def _extract_passages_hotpotqa(sample: dict) -> tuple[list[dict], list[dict]]:
    """Return (gold_passages, non_gold_passages) from HotpotQA-style sample."""
    contexts = sample.get('context', []) or []
    # Build mapping title -> list of contexts in original order
    by_title: dict[str, list[dict]] = {}
    for c in contexts:
        if not isinstance(c, dict):
            continue
        title = c.get('title')
        if title is None:
            continue
        by_title.setdefault(str(title), []).append(c)

    # Preserve supporting_facts title order (first occurrence wins)
    ordered_titles: list[str] = []
    seen_titles: set[str] = set()
    for sf in sample.get('supporting_facts', []) or []:
        if not isinstance(sf, (list, tuple)) or len(sf) < 1:
            continue
        title = str(sf[0])
        if title in seen_titles:
            continue
        seen_titles.add(title)
        ordered_titles.append(title)

    gold: list[dict] = []
    gold_texts: set[str] = set()
    gold_titles: set[str] = set()
    for title in ordered_titles:
        occurrences = by_title.get(title, [])
        if not occurrences:
            continue
        gold_titles.add(title)
        for occ in occurrences:
            sents = occ.get('sentences')
            text = " ".join(str(x) for x in sents) if isinstance(sents, list) else str(sents or '')
            text = text.strip()
            if not text or text in gold_texts:
                continue
            gold_texts.add(text)
            gold.append({"title": title, "text": text})

    non_gold: list[dict] = []
    non_gold_texts: set[str] = set()
    for c in contexts:
        if not isinstance(c, dict):
            continue
        title = str(c.get('title') or '')
        if not title:
            continue
        if title in gold_titles:
            continue
        sents = c.get('sentences')
        text = " ".join(str(x) for x in sents) if isinstance(sents, list) else str(sents or '')
        text = text.strip()
        if not text or text in gold_texts or text in non_gold_texts:
            continue
        non_gold_texts.add(text)
        non_gold.append({"title": title, "text": text})

    return gold, non_gold


def _extract_passages_musique(sample: dict) -> tuple[list[dict], list[dict]]:
    """Return (gold_passages, non_gold_passages) from MuSiQue-style sample."""
    paragraphs = sample.get('paragraphs', []) or []

    gold: list[dict] = []
    non_gold: list[dict] = []
    gold_texts: set[str] = set()
    non_gold_texts: set[str] = set()
    for p in paragraphs:
        if not isinstance(p, dict):
            continue
        title = str(p.get('title') or '')
        text = str(p.get('paragraph_text') or '').strip()
        if not text:
            continue
        if p.get('is_supporting'):
            if text in gold_texts:
                continue
            gold_texts.add(text)
            gold.append({"title": title, "text": text})
        else:
            if text in gold_texts or text in non_gold_texts:
                continue
            non_gold_texts.add(text)
            non_gold.append({"title": title, "text": text})

    return gold, non_gold


async def call_llm(prompt: str) -> str:
    """Call LLM API"""
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a precise question answering system. Give short, direct answers."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=50
    )
    return response.choices[0].message.content.strip()


async def process_question_original(
    sample: dict,
    semaphore: asyncio.Semaphore,
    prompt_mode: str,
) -> dict:
    """Answer using gold supporting passages (+ noise to fill to 5)."""
    async with semaphore:
        question = sample.get("question")
        gold_answer = sample.get("answer")

        qid = str(sample.get("_id") or sample.get("id") or "")

        if not question or gold_answer is None:
            return {
                "id": qid,
                "question": question or "",
                "gold_answer": gold_answer,
                "predicted_answer": "Insufficient information.",
                "answer_aliases": sample.get("answer_aliases", []),
                "gold_passage_count": 0,
                "noise_passage_count": 0,
                "num_injected_passages": 0,
                "success": False,
                "error": "Missing required fields: question and/or answer in input sample.",
            }

        if isinstance(sample.get('paragraphs'), list):
            gold_passages, non_gold_passages = _extract_passages_musique(sample)
        else:
            gold_passages, non_gold_passages = _extract_passages_hotpotqa(sample)

        if not gold_passages:
            return {
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": "Insufficient information.",
                "answer_aliases": sample.get("answer_aliases", []),
                "gold_passage_count": 0,
                "noise_passage_count": 0,
                "num_injected_passages": 0,
                "success": False
            }

        # Always inject exactly 5 passages: gold first, then random noise.
        injected: list[dict] = []
        injected.extend(gold_passages)

        if len(injected) > 5:
            injected = injected[:5]
        else:
            need = 5 - len(injected)
            rng = random.Random(_stable_seed(qid or question))
            if need > 0 and non_gold_passages:
                if len(non_gold_passages) <= need:
                    injected.extend(non_gold_passages)
                else:
                    injected.extend(rng.sample(non_gold_passages, need))

        passages_text = _format_passages(injected)
        prompt = build_prompt(prompt_mode, question, passages_text)
        predicted = await call_llm(prompt)
        
        return {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "answer_aliases": sample.get("answer_aliases", []),
            "gold_passage_count": len(gold_passages),
            "noise_passage_count": max(0, len(injected) - len(gold_passages)),
            "num_injected_passages": len(injected),
            "success": True
        }


async def run_experiment_original(samples: list, prompt_mode: str):
    """Run answering using gold supporting passages (plus noise to fill to 5)."""
    print("=" * 60)
    print("Perfect Retrieval with ORIGINAL PASSAGES (gold + noise)")
    print("=" * 60)
    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [process_question_original(s, semaphore, prompt_mode) for s in samples]
    results = await asyncio.gather(*tasks)
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Upper Bound Experiments for multi-hop QA')
    parser.add_argument('--dataset', type=str, default='hotpotqa',
                        choices=['hotpotqa', 'musique', '2wiki'],
                        help='Dataset to use (default: hotpotqa)')
    parser.add_argument('--data_path', type=str, default='',
                        help='Optional override for input JSON path (useful for 2wiki variants)')
    parser.add_argument('--prompt', type=str, default='naive',
                        choices=['upper', 'naive', 'final'],
                        help='Prompt mode: upper (legacy), naive (NAIVE_RAG_ANSWER_PROMPT), or final (FINAL_ANSWER_SYNTHESIS_PROMPT)')
    parser.add_argument('--max_questions', type=int, default=None,
                        help='Limit number of questions (debug / quick rerun)')
    return parser.parse_args()


async def main():
    args = parse_args()
    dataset = args.dataset.lower()
    config = DATASET_CONFIGS[dataset]
    prompt_mode = args.prompt.lower()
    
    print("=" * 80)
    print(f"Upper Bound Experiments - {dataset.upper()}")
    print("=" * 80)
    
    # Load samples
    data_path = (args.data_path or '').strip() or config['data_path']
    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise SystemExit(f"Input JSON must be a list[dict], got: {type(samples).__name__} ({data_path})")
    if args.max_questions is not None:
        samples = samples[: max(0, args.max_questions)]
    print(f"Loaded {len(samples)} samples from {data_path}")
    
    # Analyze hop distribution (for MuSiQue)
    if dataset == 'musique':
        hop_dist = {}
        for item in samples:
            # Some variants use 'id' instead of '_id'
            hop_id = str(item.get('_id') or item.get('id') or '')
            hop = hop_id.split('hop')[0] if hop_id else 'unknown'
            hop_dist[hop] = hop_dist.get(hop, 0) + 1
        print(f"Hop distribution: {hop_dist}")
    
    # Ensure Results directory exists
    Path('Results').mkdir(parents=True, exist_ok=True)

    # Run: Original Passages (Gold + Noise)
    print("\n" + "=" * 60)
    start_time = time.time()
    results_original = await run_experiment_original(samples, prompt_mode)
    time_original = time.time() - start_time
    
    # Save results
    if prompt_mode == 'final':
        out_orig = config['result_original_final']
    elif prompt_mode == 'naive':
        out_orig = config['result_original_naive']
    else:
        out_orig = config['result_original']
    with open(out_orig, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": f"{dataset.upper()} Perfect Retrieval with Original Passages (gold + noise)",
            "dataset": dataset,
            "model": MODEL,
            "prompt_mode": prompt_mode,
            "total": len(results_original),
            "time": time_original,
            "results": results_original
        }, f, ensure_ascii=False, indent=2)
    print(f"Experiment 2 completed in {time_original:.1f}s")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETED")
    print("=" * 60)
    print(f"Dataset: {dataset.upper()}")
    print(f"Results saved to:")
    print(f"  - {out_orig}")
    print(f"\nRun evaluation:")
    print(f"  python evaluate_mrqa.py {out_orig}")


if __name__ == "__main__":
    asyncio.run(main())
