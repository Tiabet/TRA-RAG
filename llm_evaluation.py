"""
LLM-based Answer Evaluation
============================
Use GPT-4o-mini to evaluate if predicted answers are correct
"""

import asyncio
import json
import os
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import time

# ============================================================
# HYPERPARAMETERS
# ============================================================
CONCURRENCY = 50  # Number of parallel LLM calls
MODEL = "openai/gpt-4o-mini"  # LLM model for evaluation
# ============================================================

# Load environment variables
load_dotenv()

# Initialize AsyncOpenAI client with ALICE API
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
    "reason": "Brief explanation of your decision (1-2 sentences)"
}}

Do not include any text before or after the JSON object."""


async def evaluate_answer_with_llm(question: str, gold_answer: str, predicted_answer: str, model: str = MODEL) -> dict:
    """
    Use LLM to evaluate if predicted answer is correct (async version)
    
    Returns:
        dict with keys: verdict (CORRECT/INCORRECT), confidence (HIGH/MEDIUM/LOW), reason
    """
    
    prompt = EVALUATION_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer
    )
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise and fair evaluator of question-answering systems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Deterministic evaluation
            max_tokens=300,
            response_format={"type": "json_object"}  # Request JSON response
        )
        
        # Parse JSON response - correct attribute access
        result_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        result = json.loads(result_text)
        
        # Validate result format
        if "verdict" not in result or "confidence" not in result or "reason" not in result:
            raise ValueError("Missing required fields in LLM response")
        
        if result["verdict"] not in ["CORRECT", "INCORRECT"]:
            raise ValueError(f"Invalid verdict: {result['verdict']}")
        
        if result["confidence"] not in ["HIGH", "MEDIUM", "LOW"]:
            raise ValueError(f"Invalid confidence: {result['confidence']}")
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing error: {e}")
        print(f"Response text: {result_text}")
        return {
            "verdict": "ERROR",
            "confidence": "LOW",
            "reason": f"Failed to parse LLM response: {str(e)}"
        }
    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
        return {
            "verdict": "ERROR",
            "confidence": "LOW",
            "reason": f"Evaluation failed: {str(e)}"
        }


def load_qa_pairs(pred_path: Path, gold_path: Path):
    """
    Load and match prediction results with gold answers
    
    Returns:
        list of dicts with keys: question_id, question, gold_answer, predicted_answer
    """
    
    # Load predicted results
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # Load gold answers
    with open(gold_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    
    # Build gold answer mapping: question -> answer
    gold_map = {}
    if isinstance(gold_data, dict):
        gold_map = gold_data
    elif isinstance(gold_data, list):
        for item in gold_data:
            question = item.get('question') or item.get('query')
            answer = item.get('answer')
            if question and answer:
                gold_map[question] = answer
    
    # Extract predictions and match with gold
    qa_pairs = []
    
    if 'results' in pred_data:
        results = pred_data['results']
    else:
        results = pred_data if isinstance(pred_data, list) else []
    
    for result in results:
        # Support both 'question' and 'query' keys
        question = result.get('question') or result.get('query')
        question_id = result.get('question_id', 'unknown')
        # Support both 'predicted_answer' and 'result' keys
        predicted_answer = result.get('predicted_answer') or result.get('result')
        
        if not question or not predicted_answer:
            continue
        
        # Find matching gold answer
        gold_answer = gold_map.get(question)
        
        if gold_answer:
            qa_pairs.append({
                'question_id': question_id,
                'question': question,
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer
            })
        else:
            print(f"⚠️ No gold answer found for question: {question[:60]}...")
    
    return qa_pairs


def evaluate_predictions(pred_path: Path, gold_path: Path,
                        model: str = MODEL, max_samples: int = None):
    """
    Evaluate all predictions using LLM (wrapper for async function)
    """
    return asyncio.run(evaluate_predictions_async(pred_path, gold_path, model, max_samples))


async def evaluate_single(qa: dict, model: str, semaphore: asyncio.Semaphore) -> dict:
    """Evaluate a single QA pair with semaphore for concurrency control."""
    async with semaphore:
        evaluation = await evaluate_answer_with_llm(
            question=qa['question'],
            gold_answer=qa['gold_answer'],
            predicted_answer=qa['predicted_answer'],
            model=model
        )
        return {
            'question_id': qa['question_id'],
            'question': qa['question'],
            'gold_answer': qa['gold_answer'],
            'predicted_answer': qa['predicted_answer'],
            'evaluation': evaluation
        }


async def evaluate_predictions_async(pred_path: Path, gold_path: Path,
                        model: str = MODEL, max_samples: int = None):
    """
    Evaluate all predictions using LLM with concurrency
    """
    
    print("="*100)
    print("🤖 LLM-Based Answer Evaluation")
    print("="*100)
    print(f"Model: {model}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Predictions: {pred_path}")
    print(f"Gold answers: {gold_path}")
    print()
    
    # Load QA pairs
    print("📁 Loading data...")
    qa_pairs = load_qa_pairs(pred_path, gold_path)
    
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
    
    print(f"✅ Loaded {len(qa_pairs)} question-answer pairs")
    print()
    
    # Evaluate each pair with concurrency
    print(f"🔍 Evaluating answers (concurrency={CONCURRENCY})...")
    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [evaluate_single(qa, model, semaphore) for qa in qa_pairs]
    
    # Use tqdm for progress
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        result = await coro
        results.append(result)
    
    # Count results
    correct_count = 0
    incorrect_count = 0
    error_count = 0
    
    for result in results:
        evaluation = result['evaluation']
        if evaluation['verdict'] == 'CORRECT':
            correct_count += 1
        elif evaluation['verdict'] == 'INCORRECT':
            incorrect_count += 1
        else:  # ERROR
            error_count += 1
    
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0
    
    print()
    print("="*60)
    print("📊 LLM EVALUATION RESULTS")
    print("="*60)
    print(f"Total: {total}")
    print(f"✅ Correct: {correct_count} ({correct_count/total*100:.1f}%)")
    print(f"❌ Incorrect: {incorrect_count} ({incorrect_count/total*100:.1f}%)")
    if error_count > 0:
        print(f"⚠️ Errors: {error_count}")
    print(f"🎯 LLM Accuracy: {accuracy:.3f} ({correct_count}/{total})")
    print("="*60)
    
    # Save detailed results
    output_data = {
        'config': {
            'model': model,
            'predictions_file': str(pred_path),
            'gold_file': str(gold_path),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        },
        'summary': {
            'total': total,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'errors': error_count,
            'accuracy': accuracy
        },
        'results': results
    }
    
    output_path = pred_path.parent / f"llm_eval_{pred_path.name}"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n💾 Detailed evaluation results saved to: {output_path}")
    
    return {'accuracy': accuracy, 'correct': correct_count, 'total': total}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-based answer evaluation")
    parser.add_argument('--pred', type=Path, default=Path('Results/NaiveRAG/NaiveRAG_passage_QD_musique.json'),
                       help='Path to predictions file')
    parser.add_argument('--gold', type=Path, default=Path('MuSiQue/qa.json'),
                       help='Path to gold answers file')
    parser.add_argument('--model', type=str, default='openai/gpt-4o-mini',
                       help='OpenAI model to use for evaluation')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    evaluate_predictions(
        pred_path=args.pred,
        gold_path=args.gold,
        model=args.model,
        max_samples=args.max_samples
    )
