#!/usr/bin/env python
"""
MRQA Official Evaluation Metrics
=================================
MRQA 공식 평가 메트릭 (Exact Match, F1)을 사용한 평가 스크립트

Usage:
    python evaluate_mrqa.py Results/test_pipeline_v3_original_200_results.json
    python evaluate_mrqa.py Results/test_new_pipeline_200_results_v2.json
"""
import json
import re
import string
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict


# ============================================================
# MRQA Official Normalization (from MRQA eval script)
# ============================================================
def normalize_answer(s: str) -> str:
    """
    Lower text and remove punctuation, articles and extra whitespace.
    Based on MRQA official evaluation script.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ============================================================
# MRQA Official Metrics
# ============================================================
def compute_exact_match(gold: str, predicted: str) -> float:
    """Compute Exact Match score."""
    return 1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0


def compute_accuracy(gold: str, predicted: str) -> float:
    """
    Compute Accuracy score.
    Returns 1.0 if normalized gold answer is contained in normalized predicted answer.
    """
    norm_gold = normalize_answer(gold)
    norm_pred = normalize_answer(predicted)
    
    if not norm_gold or not norm_pred:
        return 0.0
    
    return 1.0 if norm_gold in norm_pred else 0.0


def compute_f1(gold: str, predicted: str) -> Tuple[float, float, float]:
    """
    Compute F1 score.
    
    Returns:
        Tuple of (F1, Precision, Recall)
    """
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    
    # Handle empty cases
    if len(gold_tokens) == 0 or len(predicted_tokens) == 0:
        # Both empty = perfect match
        if len(gold_tokens) == len(predicted_tokens) == 0:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0
    
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0, 0.0, 0.0
    
    precision = num_same / len(predicted_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1, precision, recall


def compute_metrics_with_aliases(
    gold_answers: List[str], 
    predicted: str
) -> Dict[str, float]:
    """
    Compute metrics considering multiple gold answers (aliases).
    Takes the maximum score across all gold answers.
    
    Args:
        gold_answers: List of acceptable gold answers
        predicted: Predicted answer string
        
    Returns:
        Dict with EM, Accuracy, F1, Precision, Recall (best across all gold answers)
    """
    best_em = 0.0
    best_accuracy = 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    for gold in gold_answers:
        em = compute_exact_match(gold, predicted)
        accuracy = compute_accuracy(gold, predicted)
        f1, precision, recall = compute_f1(gold, predicted)
        
        if em > best_em:
            best_em = em
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
    
    return {
        "exact_match": best_em,
        "accuracy": best_accuracy,
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall
    }


# ============================================================
# Result File Loading
# ============================================================
def load_results(file_path: Path) -> List[Dict]:
    """Load results from JSON file."""
    with file_path.open(encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle different result formats
    if "results" in data:
        return data["results"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown result format in {file_path}")


def extract_gold_answers(item: Dict) -> List[str]:
    """
    Extract gold answers from result item.
    Handles both single answer and answer aliases.
    """
    gold_answers = []
    
    # Primary gold answer
    if "gold_answer" in item:
        gold_answers.append(item["gold_answer"])
    elif "answer" in item:
        gold_answers.append(item["answer"])
    
    # Answer aliases (if available)
    if "answer_aliases" in item and item["answer_aliases"]:
        gold_answers.extend(item["answer_aliases"])
    
    return gold_answers


def extract_predicted_answer(item: Dict) -> str:
    """Extract predicted answer from result item."""
    if "predicted_answer" in item:
        return item["predicted_answer"]
    elif "prediction" in item:
        return item["prediction"]
    elif "final_answer" in item:
        return item["final_answer"]
    return ""


# ============================================================
# Main Evaluation
# ============================================================
def evaluate(file_path: Path, verbose: bool = False) -> Dict[str, float]:
    """
    Evaluate a result file using MRQA official metrics.
    
    Args:
        file_path: Path to result JSON file
        verbose: Print per-example results
        
    Returns:
        Dict with aggregated metrics
    """
    print(f"📂 Loading: {file_path}")
    results = load_results(file_path)
    print(f"   Total examples: {len(results)}")
    
    # Accumulate scores
    total_em = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    
    # Additional stats
    insufficient_count = 0
    perfect_match_count = 0
    accuracy_match_count = 0
    
    example_results = []
    
    for i, item in enumerate(results):
        gold_answers = extract_gold_answers(item)
        predicted = extract_predicted_answer(item)
        
        # Track "Insufficient information" answers
        if "Insufficient information" in predicted or not predicted:
            insufficient_count += 1
        
        # Compute metrics
        metrics = compute_metrics_with_aliases(gold_answers, predicted)
        
        total_em += metrics["exact_match"]
        total_accuracy += metrics["accuracy"]
        total_f1 += metrics["f1"]
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        
        if metrics["exact_match"] == 1.0:
            perfect_match_count += 1
        if metrics["accuracy"] == 1.0:
            accuracy_match_count += 1
        
        example_results.append({
            "id": item.get("id", i),
            "question": item.get("question", ""),
            "gold": gold_answers[0] if gold_answers else "",
            "predicted": predicted,
            **metrics
        })
        
        if verbose:
            status = "✓" if metrics["exact_match"] == 1.0 else ("~" if metrics["f1"] > 0.5 else "✗")
            print(f"  [{status}] Q{i+1}: EM={metrics['exact_match']:.0f} F1={metrics['f1']:.3f} | Gold: {gold_answers[0][:30]}... | Pred: {predicted[:30]}...")
    
    # Compute averages
    n = len(results)
    avg_metrics = {
        "exact_match": total_em / n if n else 0,
        "accuracy": total_accuracy / n if n else 0,
        "f1": total_f1 / n if n else 0,
        "precision": total_precision / n if n else 0,
        "recall": total_recall / n if n else 0,
        "total": n,
        "perfect_matches": int(total_em),
        "accuracy_matches": int(total_accuracy),
        "insufficient_answers": insufficient_count
    }
    
    return avg_metrics, example_results


def print_results(metrics: Dict[str, float], file_path: Path):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"📊 MRQA Official Evaluation Results")
    print(f"{'='*60}")
    print(f"File: {file_path.name}")
    print(f"{'='*60}")
    print(f"Exact Match (EM) : {metrics['exact_match']:.3f} ({metrics['perfect_matches']}/{metrics['total']})")
    print(f"Accuracy         : {metrics['accuracy']:.3f} ({metrics['accuracy_matches']}/{metrics['total']})")
    print(f"F1 Score         : {metrics['f1']:.3f}")
    print(f"Precision        : {metrics['precision']:.3f}")
    print(f"Recall           : {metrics['recall']:.3f}")
    print(f"{'='*60}")
    print(f"Insufficient answers: {metrics['insufficient_answers']}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="MRQA Official Evaluation")
    parser.add_argument("file", type=str, nargs="?", 
                        default="Results/upper_bound_original_results.json",
                        help="Path to result JSON file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-example results")
    parser.add_argument("--compare", type=str, nargs="+",
                        help="Compare multiple result files")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple files
        print(f"\n{'='*90}")
        print("📊 Comparison of Multiple Results")
        print(f"{'='*90}")
        print(f"{'File':<45} {'EM':>8} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
        print(f"{'-'*90}")
        
        for file_path in args.compare:
            path = Path(file_path)
            if path.exists():
                metrics, _ = evaluate(path, verbose=False)
                print(f"{path.name:<45} {metrics['exact_match']:>8.3f} {metrics['accuracy']:>8.3f} {metrics['f1']:>8.3f} {metrics['precision']:>8.3f} {metrics['recall']:>8.3f}")
            else:
                print(f"{path.name:<45} {'FILE NOT FOUND':>44}")
        
        print(f"{'='*90}\n")
    else:
        # Single file evaluation
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return
        
        metrics, example_results = evaluate(file_path, verbose=args.verbose)
        print_results(metrics, file_path)


if __name__ == "__main__":
    main()
