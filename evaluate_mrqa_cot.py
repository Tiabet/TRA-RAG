#!/usr/bin/env python
"""\
MRQA Official Evaluation Metrics (CoT output adapter)
===================================================

`evaluate_mrqa.py`와 동일한 MRQA 공식 메트릭(Exact Match, Accuracy, F1)을 사용하되,
모델 출력이 CoT 형태(예: "Thought: ...\nAnswer: ...")인 경우를 위해
"Answer:" 이후의 답변만 추출해서 평가합니다.

Usage:
    python evaluate_mrqa_cot.py Results/test_musique_v11_ragprompt_results.json Results/NaiveRAG/NaiveRAG_passage_QD_cot_musique.json
    python evaluate_mrqa_cot.py Results/test_hotpot_v11_ragcot_results.json
    python evaluate_mrqa_cot.py --compare Results/a.json Results/b.json
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================
# MRQA Official Normalization (from MRQA eval script)
# ============================================================
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

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
    return 1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0


def compute_accuracy(gold: str, predicted: str) -> float:
    """Returns 1.0 if normalized gold answer is contained in normalized predicted answer."""
    norm_gold = normalize_answer(gold)
    norm_pred = normalize_answer(predicted)

    if not norm_gold or not norm_pred:
        return 0.0

    return 1.0 if norm_gold in norm_pred else 0.0


def compute_f1(gold: str, predicted: str) -> Tuple[float, float, float]:
    """Returns (F1, Precision, Recall)."""
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()

    if len(gold_tokens) == 0 or len(predicted_tokens) == 0:
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


def compute_metrics_with_aliases(gold_answers: List[str], predicted: str) -> Dict[str, float]:
    """Take the maximum score across all gold answers (aliases)."""
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
        'exact_match': best_em,
        'accuracy': best_accuracy,
        'f1': best_f1,
        'precision': best_precision,
        'recall': best_recall,
    }


# ============================================================
# Result File Loading
# ============================================================
def load_results(file_path: Path) -> List[Dict]:
    with file_path.open(encoding='utf-8') as f:
        data = json.load(f)

    if 'results' in data:
        return data['results']
    if isinstance(data, list):
        return data
    raise ValueError(f'Unknown result format in {file_path}')


def extract_gold_answers(item: Dict) -> List[str]:
    gold_answers: List[str] = []

    if 'gold_answer' in item:
        gold_answers.append(item['gold_answer'])
    elif 'answer' in item:
        gold_answers.append(item['answer'])

    if 'answer_aliases' in item and item['answer_aliases']:
        gold_answers.extend(item['answer_aliases'])

    return gold_answers


def _extract_after_answer_marker(text: str) -> str:
    """Extract only the portion after the last 'Answer:' marker.

    - If there is no 'Answer:' marker, returns the original text.
    - Keeps everything after the marker but trims whitespace.
    - If the extracted part is multi-line, keeps it as-is (MRQA normalization handles spacing).
    """
    if not text:
        return ''

    matches = list(re.finditer(r'(?i)\bAnswer\s*:\s*', text))
    if not matches:
        return text.strip()

    last = matches[-1]
    extracted = text[last.end():].strip()
    return extracted


def extract_predicted_answer(item: Dict) -> str:
    if 'predicted_answer' in item:
        predicted = item['predicted_answer']
    elif 'prediction' in item:
        predicted = item['prediction']
    elif 'final_answer' in item:
        predicted = item['final_answer']
    else:
        predicted = ''

    return _extract_after_answer_marker(predicted)


# ============================================================
# Main Evaluation
# ============================================================
def evaluate(file_path: Path, verbose: bool = False) -> Tuple[Dict[str, float], List[Dict]]:
    print(f'📂 Loading: {file_path}')
    results = load_results(file_path)
    print(f'   Total examples: {len(results)}')

    total_em = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0

    insufficient_count = 0
    perfect_match_count = 0
    accuracy_match_count = 0

    example_results: List[Dict] = []

    for i, item in enumerate(results):
        gold_answers = extract_gold_answers(item)
        predicted = extract_predicted_answer(item)

        if 'Insufficient information' in (predicted or '') or not predicted:
            insufficient_count += 1

        metrics = compute_metrics_with_aliases(gold_answers, predicted)

        total_em += metrics['exact_match']
        total_accuracy += metrics['accuracy']
        total_f1 += metrics['f1']
        total_precision += metrics['precision']
        total_recall += metrics['recall']

        if metrics['exact_match'] == 1.0:
            perfect_match_count += 1
        if metrics['accuracy'] == 1.0:
            accuracy_match_count += 1

        example_results.append({
            'id': item.get('id', i),
            'question': item.get('question', ''),
            'gold': gold_answers[0] if gold_answers else '',
            'predicted': predicted,
            **metrics,
        })

        if verbose:
            status = '✓' if metrics['exact_match'] == 1.0 else ('~' if metrics['f1'] > 0.5 else '✗')
            g0 = gold_answers[0] if gold_answers else ''
            print(
                f"  [{status}] Q{i+1}: EM={metrics['exact_match']:.0f} F1={metrics['f1']:.3f} | "
                f"Gold: {g0[:30]}... | Pred: {predicted[:30]}..."
            )

    n = len(results)
    avg_metrics = {
        'exact_match': total_em / n if n else 0,
        'accuracy': total_accuracy / n if n else 0,
        'f1': total_f1 / n if n else 0,
        'precision': total_precision / n if n else 0,
        'recall': total_recall / n if n else 0,
        'total': n,
        'perfect_matches': int(total_em),
        'accuracy_matches': int(total_accuracy),
        'insufficient_answers': insufficient_count,
    }

    return avg_metrics, example_results


def print_results(metrics: Dict[str, float], file_path: Path):
    print(f"\n{'='*60}")
    print('📊 MRQA Official Evaluation Results (CoT adapter)')
    print(f"{'='*60}")
    print(f'File: {file_path.name}')
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
    parser = argparse.ArgumentParser(description='MRQA Official Evaluation (CoT output adapter)')
    parser.add_argument(
        'file',
        type=str,
        nargs='?',
        default='Results/test_musique_v11_ragprompt_results.json',
        help='Path to result JSON file',
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Print per-example results')
    parser.add_argument('--compare', type=str, nargs='+', help='Compare multiple result files')

    args = parser.parse_args()

    if args.compare:
        print(f"\n{'='*90}")
        print('📊 Comparison of Multiple Results (CoT adapter)')
        print(f"{'='*90}")
        print(f"{'File':<45} {'EM':>8} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
        print(f"{'-'*90}")

        for file_path in args.compare:
            path = Path(file_path)
            if path.exists():
                metrics, _ = evaluate(path, verbose=False)
                print(
                    f"{path.name:<45} {metrics['exact_match']:>8.3f} {metrics['accuracy']:>8.3f} "
                    f"{metrics['f1']:>8.3f} {metrics['precision']:>8.3f} {metrics['recall']:>8.3f}"
                )
            else:
                print(f"{path.name:<45} {'FILE NOT FOUND':>44}")

        print(f"{'='*90}\n")
        return

    file_path = Path(args.file)
    if not file_path.exists():
        print(f'❌ File not found: {file_path}')
        return

    metrics, _example_results = evaluate(file_path, verbose=args.verbose)
    print_results(metrics, file_path)


if __name__ == '__main__':
    main()
