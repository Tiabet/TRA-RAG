"""
Upper Bound Experiments
========================
Experiment 1: Perfect Retrieval with Metadata
Experiment 2: Perfect Retrieval with Original Passages

Usage:
    python upper_bound_experiment.py                     # Default: HotpotQA
    python upper_bound_experiment.py --dataset hotpotqa  # HotpotQA
    python upper_bound_experiment.py --dataset musique   # MuSiQue

Optional:
        python upper_bound_experiment.py --dataset hotpotqa --prompt final
            - Uses the repo's FINAL_ANSWER_SYNTHESIS_PROMPT (same as current pipeline)
"""

import json
import asyncio
import os
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Prompt.subquestion_answering_prompt import FINAL_ANSWER_SYNTHESIS_PROMPT

load_dotenv()

# ==================== HYPERPARAMETERS ====================
CONCURRENCY = 50
MODEL = "openai/gpt-4o-mini"
# =========================================================

# Dataset configurations
DATASET_CONFIGS = {
    'hotpotqa': {
        'data_path': 'HotpotQA/hotpotqa_sample_200.json',
        'db_path': 'HotpotQA/metadata_v3.db',
        'result_metadata': 'Results/upper_bound_metadata_results.json',
        'result_original': 'Results/upper_bound_original_results.json',
        'result_metadata_final': 'Results/upper_bound_metadata_results_finalprompt.json',
        'result_original_final': 'Results/upper_bound_original_results_finalprompt.json',
    },
    'musique': {
        'data_path': 'MuSiQue/musique_sample_200.json',
        'db_path': 'MuSiQue/metadata_v3.db',
        'result_metadata': 'Results/upper_bound_musique_metadata_results.json',
        'result_original': 'Results/upper_bound_musique_original_results.json',
        'result_metadata_final': 'Results/upper_bound_musique_metadata_results_finalprompt.json',
        'result_original_final': 'Results/upper_bound_musique_original_results_finalprompt.json',
    }
}

# Initialize client
client = AsyncOpenAI(
    api_key=os.getenv('ALICE_OPENAI_KEY'),
    base_url=os.getenv('ALICE_CHAT_URL')
)

# Prompt template (same constraints as final answer prompt)
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


def build_final_synthesis_prompt(question: str, passages_text: str) -> str:
    """Use the same final-answer prompt template as the current pipeline."""
    subquestion_chain = "N/A (Gold supporting facts provided directly)"
    prompt = FINAL_ANSWER_SYNTHESIS_PROMPT.replace("{{main_question}}", question)
    prompt = prompt.replace("{{subquestion_chain}}", subquestion_chain)
    prompt = prompt.replace("{{passages}}", passages_text if passages_text else "No passages.")
    return prompt


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


async def process_question_metadata(
    sample: dict,
    metadata_db: dict,
    semaphore: asyncio.Semaphore,
    prompt_mode: str,
) -> dict:
    """Experiment 1: Answer using metadata for supporting facts"""
    async with semaphore:
        question = sample["question"]
        gold_answer = sample["answer"]
        qid = sample.get("_id", "")
        
        # Get supporting fact titles
        sf_titles = list(set([sf[0] for sf in sample["supporting_facts"]]))
        
        # Get metadata for each title
        passages_text = ""
        found_titles = []
        for i, title in enumerate(sf_titles, 1):
            if title in metadata_db:
                metadata = metadata_db[title]
                passages_text += f"[{i}] {title}\n"
                passages_text += json.dumps(metadata, ensure_ascii=False, indent=2)
                passages_text += "\n\n"
                found_titles.append(title)
        
        if not passages_text:
            return {
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": "Insufficient information.",
                "answer_aliases": sample.get("answer_aliases", []),
                "sf_titles": sf_titles,
                "found_titles": found_titles,
                "success": False
            }
        
        if prompt_mode == 'final':
            prompt = build_final_synthesis_prompt(question, passages_text)
        else:
            prompt = UPPER_BOUND_PROMPT.format(passages=passages_text, question=question)
        predicted = await call_llm(prompt)
        
        return {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "answer_aliases": sample.get("answer_aliases", []),
            "sf_titles": sf_titles,
            "found_titles": found_titles,
            "success": True
        }


async def process_question_original(
    sample: dict,
    semaphore: asyncio.Semaphore,
    prompt_mode: str,
) -> dict:
    """Experiment 2: Answer using original passages from hotpot.jsonl"""
    async with semaphore:
        question = sample["question"]
        gold_answer = sample["answer"]
        qid = sample.get("_id", "")
        
        # Get supporting fact titles and sentence indices
        sf_info = {}  # title -> set of sentence indices
        for sf in sample["supporting_facts"]:
            title, sent_idx = sf[0], sf[1]
            if title not in sf_info:
                sf_info[title] = set()
            sf_info[title].add(sent_idx)
        
        # Get original passages from context
        passages_text = ""
        context_dict = {ctx[0]: ctx[1] for ctx in sample["context"]}
        
        for i, (title, sent_indices) in enumerate(sf_info.items(), 1):
            if title in context_dict:
                sentences = context_dict[title]
                # Get all sentences from the passage (not just supporting fact sentences)
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
                "success": False
            }
        
        if prompt_mode == 'final':
            prompt = build_final_synthesis_prompt(question, passages_text)
        else:
            prompt = UPPER_BOUND_PROMPT.format(passages=passages_text, question=question)
        predicted = await call_llm(prompt)
        
        return {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "answer_aliases": sample.get("answer_aliases", []),
            "sf_titles": list(sf_info.keys()),
            "success": True
        }


async def run_experiment_metadata(samples: list, metadata_db: dict, prompt_mode: str):
    """Run Experiment 1: Metadata-based answering"""
    print("=" * 60)
    print("Experiment 1: Perfect Retrieval with METADATA")
    print("=" * 60)
    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [process_question_metadata(s, metadata_db, semaphore, prompt_mode) for s in samples]
    results = await asyncio.gather(*tasks)
    
    return results


async def run_experiment_original(samples: list, prompt_mode: str):
    """Run Experiment 2: Original passage-based answering"""
    print("=" * 60)
    print("Experiment 2: Perfect Retrieval with ORIGINAL PASSAGES")
    print("=" * 60)
    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [process_question_original(s, semaphore, prompt_mode) for s in samples]
    results = await asyncio.gather(*tasks)
    
    return results


def load_metadata_db(db_path: str):
    """Load metadata from database"""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    metadata_db = {}

    # Support both current schema (doc_id/source_title/entity_title/metadata_json)
    # and legacy schema (title/metadata_json).
    try:
        cursor.execute("SELECT source_title, entity_title, metadata_json FROM metadata")
        rows = cursor.fetchall()
        for source_title, entity_title, metadata_json in rows:
            meta = json.loads(metadata_json)
            # supporting_facts titles correspond to source_title in the dataset
            if source_title and source_title not in metadata_db:
                metadata_db[source_title] = meta
            # also index by entity_title as a fallback
            if entity_title and entity_title not in metadata_db:
                metadata_db[entity_title] = meta
    except Exception:
        cursor.execute("SELECT title, metadata_json FROM metadata")
        rows = cursor.fetchall()
        for title, metadata_json in rows:
            metadata_db[title] = json.loads(metadata_json)
    
    conn.close()
    print(f"Loaded {len(metadata_db)} metadata entries from {db_path}")
    return metadata_db


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Upper Bound Experiments for multi-hop QA')
    parser.add_argument('--dataset', type=str, default='hotpotqa',
                        choices=['hotpotqa', 'musique'],
                        help='Dataset to use (default: hotpotqa)')
    parser.add_argument('--prompt', type=str, default='upper',
                        choices=['upper', 'final'],
                        help='Prompt mode: upper (legacy upper-bound prompt) or final (FINAL_ANSWER_SYNTHESIS_PROMPT)')
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
    with open(config['data_path'], "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from {config['data_path']}")
    
    # Analyze hop distribution (for MuSiQue)
    if dataset == 'musique':
        hop_dist = {}
        for item in samples:
            hop = item['_id'].split('hop')[0]
            hop_dist[hop] = hop_dist.get(hop, 0) + 1
        print(f"Hop distribution: {hop_dist}")
    
    # Load metadata
    metadata_db = load_metadata_db(config['db_path'])
    
    # Check how many supporting facts are in metadata
    total_sf = 0
    found_sf = 0
    for sample in samples:
        sf_titles = set([sf[0] for sf in sample["supporting_facts"]])
        total_sf += len(sf_titles)
        for title in sf_titles:
            if title in metadata_db:
                found_sf += 1
    print(f"Supporting facts coverage: {found_sf}/{total_sf} ({found_sf/total_sf*100:.1f}%)")
    
    # Ensure Results directory exists
    Path('Results').mkdir(parents=True, exist_ok=True)
    
    # Run Experiment 1: Metadata
    print("\n" + "=" * 60)
    start_time = time.time()
    results_metadata = await run_experiment_metadata(samples, metadata_db, prompt_mode)
    time_metadata = time.time() - start_time
    
    # Save results
    out_meta = config['result_metadata_final'] if prompt_mode == 'final' else config['result_metadata']
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": f"{dataset.upper()} Perfect Retrieval with Metadata",
            "dataset": dataset,
            "model": MODEL,
            "prompt_mode": prompt_mode,
            "total": len(results_metadata),
            "time": time_metadata,
            "results": results_metadata
        }, f, ensure_ascii=False, indent=2)
    print(f"Experiment 1 completed in {time_metadata:.1f}s")
    
    # Run Experiment 2: Original Passages
    print("\n" + "=" * 60)
    start_time = time.time()
    results_original = await run_experiment_original(samples, prompt_mode)
    time_original = time.time() - start_time
    
    # Save results
    out_orig = config['result_original_final'] if prompt_mode == 'final' else config['result_original']
    with open(out_orig, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": f"{dataset.upper()} Perfect Retrieval with Original Passages",
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
    print(f"  - {out_meta}")
    print(f"  - {out_orig}")
    print(f"\nRun evaluation:")
    print(f"  python evaluate_mrqa.py {out_meta}")
    print(f"  python evaluate_mrqa.py {out_orig}")


if __name__ == "__main__":
    asyncio.run(main())
