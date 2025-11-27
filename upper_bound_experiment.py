"""
Upper Bound Experiments
========================
Experiment 1: Perfect Retrieval with Metadata
Experiment 2: Perfect Retrieval with Original Passages
"""

import json
import asyncio
import os
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ==================== HYPERPARAMETERS ====================
CONCURRENCY = 10
MODEL = "openai/gpt-4o-mini"
# =========================================================

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


async def call_llm(prompt: str) -> str:
    """Call LLM API"""
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50
    )
    return response.choices[0].message.content.strip()


async def process_question_metadata(sample: dict, metadata_db: dict, semaphore: asyncio.Semaphore) -> dict:
    """Experiment 1: Answer using metadata for supporting facts"""
    async with semaphore:
        question = sample["question"]
        gold_answer = sample["answer"]
        
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
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": "Insufficient information.",
                "sf_titles": sf_titles,
                "found_titles": found_titles,
                "success": False
            }
        
        prompt = UPPER_BOUND_PROMPT.format(passages=passages_text, question=question)
        predicted = await call_llm(prompt)
        
        return {
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "sf_titles": sf_titles,
            "found_titles": found_titles,
            "success": True
        }


async def process_question_original(sample: dict, semaphore: asyncio.Semaphore) -> dict:
    """Experiment 2: Answer using original passages from hotpot.jsonl"""
    async with semaphore:
        question = sample["question"]
        gold_answer = sample["answer"]
        
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
                full_passage = "".join(sentences)
                passages_text += f"[{i}] {title}\n{full_passage}\n\n"
        
        if not passages_text:
            return {
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": "Insufficient information.",
                "sf_titles": list(sf_info.keys()),
                "success": False
            }
        
        prompt = UPPER_BOUND_PROMPT.format(passages=passages_text, question=question)
        predicted = await call_llm(prompt)
        
        return {
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "sf_titles": list(sf_info.keys()),
            "success": True
        }


async def run_experiment_metadata(samples: list, metadata_db: dict):
    """Run Experiment 1: Metadata-based answering"""
    print("=" * 60)
    print("Experiment 1: Perfect Retrieval with METADATA")
    print("=" * 60)
    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [process_question_metadata(s, metadata_db, semaphore) for s in samples]
    results = await asyncio.gather(*tasks)
    
    return results


async def run_experiment_original(samples: list):
    """Run Experiment 2: Original passage-based answering"""
    print("=" * 60)
    print("Experiment 2: Perfect Retrieval with ORIGINAL PASSAGES")
    print("=" * 60)
    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [process_question_original(s, semaphore) for s in samples]
    results = await asyncio.gather(*tasks)
    
    return results


def load_metadata_db():
    """Load metadata from database"""
    import sqlite3
    
    db_path = "HotpotQA/metadata_v3.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT title, metadata_json FROM metadata")
    rows = cursor.fetchall()
    
    metadata_db = {}
    for title, metadata_json in rows:
        metadata_db[title] = json.loads(metadata_json)
    
    conn.close()
    print(f"Loaded {len(metadata_db)} metadata entries")
    return metadata_db


async def main():
    # Load samples
    with open("HotpotQA/hotpotqa_sample_200.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")
    
    # Load metadata
    metadata_db = load_metadata_db()
    
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
    
    # Run Experiment 1: Metadata
    print("\n" + "=" * 60)
    start_time = time.time()
    results_metadata = await run_experiment_metadata(samples, metadata_db)
    time_metadata = time.time() - start_time
    
    # Save results
    with open("Results/upper_bound_metadata_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Perfect Retrieval with Metadata",
            "model": MODEL,
            "total": len(results_metadata),
            "time": time_metadata,
            "results": results_metadata
        }, f, ensure_ascii=False, indent=2)
    print(f"Experiment 1 completed in {time_metadata:.1f}s")
    
    # Run Experiment 2: Original Passages
    print("\n" + "=" * 60)
    start_time = time.time()
    results_original = await run_experiment_original(samples)
    time_original = time.time() - start_time
    
    # Save results
    with open("Results/upper_bound_original_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Perfect Retrieval with Original Passages",
            "model": MODEL,
            "total": len(results_original),
            "time": time_original,
            "results": results_original
        }, f, ensure_ascii=False, indent=2)
    print(f"Experiment 2 completed in {time_original:.1f}s")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETED")
    print("=" * 60)
    print(f"Results saved to:")
    print(f"  - Results/upper_bound_metadata_results.json")
    print(f"  - Results/upper_bound_original_results.json")
    print(f"\nRun llm_evaluation.py on these results to get accuracy scores.")


if __name__ == "__main__":
    asyncio.run(main())
