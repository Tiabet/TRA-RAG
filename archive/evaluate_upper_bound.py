"""
Evaluate Upper Bound Experiment Results
"""

import asyncio
import json
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# HYPERPARAMETERS
# ============================================================
CONCURRENCY = 50
MODEL = "openai/gpt-4o-mini"
# ============================================================

client = AsyncOpenAI(
    api_key=os.getenv('ALICE_OPENAI_KEY'),
    base_url=os.getenv('ALICE_CHAT_URL')
)

EVALUATION_PROMPT = """You are an expert evaluator for question-answering systems. Your task is to determine if a predicted answer is correct given a question and the gold (correct) answer.

**Question:** {question}

**Gold Answer:** {gold_answer}

**Predicted Answer:** {predicted_answer}

**Evaluation Guidelines:**
1. The predicted answer is CORRECT if it contains the essential information from the gold answer, even if worded differently
2. The predicted answer is CORRECT if it provides the same factual information, even with additional context
3. The predicted answer is INCORRECT if it contradicts the gold answer or provides wrong information
4. The predicted answer is INCORRECT if it says "Insufficient information" or similar when a clear answer exists
5. Minor differences in wording, formatting, or additional context are acceptable as long as the core fact is correct

**Your Task:**
Evaluate whether the predicted answer is correct or incorrect. Respond with ONLY a JSON object in this exact format:
{{
    "verdict": "CORRECT" or "INCORRECT",
    "confidence": "HIGH" or "MEDIUM" or "LOW",
    "reason": "Brief explanation"
}}
"""


async def evaluate_single(item: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        prompt = EVALUATION_PROMPT.format(
            question=item["question"],
            gold_answer=item["gold_answer"],
            predicted_answer=item["predicted_answer"]
        )
        
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            return {
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "predicted_answer": item["predicted_answer"],
                "verdict": result.get("verdict", "ERROR"),
                "confidence": result.get("confidence", "LOW"),
                "reason": result.get("reason", "")
            }
        except Exception as e:
            return {
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "predicted_answer": item["predicted_answer"],
                "verdict": "ERROR",
                "confidence": "LOW",
                "reason": str(e)
            }


async def evaluate_results(input_file: str, output_file: str):
    print(f"\nEvaluating: {input_file}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data["results"]
    print(f"Total: {len(results)} items")
    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [evaluate_single(r, semaphore) for r in results]
    evaluations = await asyncio.gather(*tasks)
    
    # Calculate stats
    correct = sum(1 for e in evaluations if e["verdict"] == "CORRECT")
    incorrect = sum(1 for e in evaluations if e["verdict"] == "INCORRECT")
    errors = sum(1 for e in evaluations if e["verdict"] == "ERROR")
    
    accuracy = correct / len(evaluations) * 100
    
    print(f"Results: {correct} correct, {incorrect} incorrect, {errors} errors")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # Save
    output_data = {
        "experiment": data["experiment"],
        "model": data["model"],
        "total": len(evaluations),
        "correct": correct,
        "incorrect": incorrect,
        "errors": errors,
        "accuracy": accuracy,
        "evaluations": evaluations
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to: {output_file}")
    return accuracy


async def main():
    print("=" * 60)
    print("Upper Bound Experiment Evaluation")
    print("=" * 60)
    
    # Evaluate Experiment 1: Metadata
    acc1 = await evaluate_results(
        "Results/upper_bound_metadata_results.json",
        "Results/upper_bound_metadata_evaluation.json"
    )
    
    # Evaluate Experiment 2: Original Passages
    acc2 = await evaluate_results(
        "Results/upper_bound_original_results.json",
        "Results/upper_bound_original_evaluation.json"
    )
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Experiment 1 (Metadata):         {acc1:.1f}%")
    print(f"Experiment 2 (Original Passages): {acc2:.1f}%")
    print(f"Current Pipeline:                 81.5%")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
