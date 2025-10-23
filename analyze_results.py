"""
Evaluate Results from 200 QA Test
==================================
Analyzes both retrieval recall and answer quality metrics.
"""

import json
from collections import defaultdict
from typing import Dict, List

def calculate_exact_match(predicted: str, gold: str) -> bool:
    """Calculate exact match with normalization."""
    def normalize(text):
        if not text:
            return ""
        text = str(text).lower().strip()
        # Remove articles
        for article in ['the', 'a', 'an']:
            text = text.replace(f' {article} ', ' ')
        # Remove punctuation
        import string
        text = ''.join(c if c not in string.punctuation else ' ' for c in text)
        return ' '.join(text.split())
    
    return normalize(predicted) == normalize(gold)


def calculate_token_f1(predicted: str, gold: str) -> float:
    """Calculate token-level F1 score."""
    def get_tokens(text):
        if not text:
            return set()
        text = str(text).lower().strip()
        import string
        text = ''.join(c if c not in string.punctuation else ' ' for c in text)
        return set(text.split())
    
    pred_tokens = get_tokens(predicted)
    gold_tokens = get_tokens(gold)
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def calculate_retrieval_recall(result: Dict) -> bool:
    """Check if all gold supporting facts are retrieved."""
    gold_facts = result.get('gold_supporting_facts', [])
    retrieved_titles = result.get('retrieved_passages', {}).get('titles', [])
    
    if not gold_facts:
        return True  # No gold facts to retrieve
    
    gold_titles = set(fact[0] for fact in gold_facts)
    retrieved_set = set(retrieved_titles)
    
    return gold_titles.issubset(retrieved_set)


def analyze_results(results_file: str):
    """Analyze results from test run."""
    
    print("=" * 80)
    print("EVALUATION RESULTS ANALYSIS")
    print("=" * 80)
    
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    print(f"\nTotal Questions: {len(results)}")
    
    # Initialize metrics
    exact_matches = []
    token_f1_scores = []
    retrieval_recalls = []
    
    # By question type
    by_type = defaultdict(lambda: {
        'count': 0,
        'exact_match': 0,
        'avg_f1': [],
        'retrieval_recall': 0
    })
    
    # By question level
    by_level = defaultdict(lambda: {
        'count': 0,
        'exact_match': 0,
        'avg_f1': [],
        'retrieval_recall': 0
    })
    
    # Detailed failure analysis
    failed_questions = []
    insufficient_info_count = 0
    
    # Process each result
    for result in results:
        predicted = result.get('predicted_answer', '')
        gold = result.get('gold_answer', '')
        q_type = result.get('question_type', 'unknown')
        q_level = result.get('question_level', 'unknown')
        
        # Calculate metrics
        em = calculate_exact_match(predicted, gold)
        f1 = calculate_token_f1(predicted, gold)
        recall = calculate_retrieval_recall(result)
        
        exact_matches.append(em)
        token_f1_scores.append(f1)
        retrieval_recalls.append(recall)
        
        # By type
        by_type[q_type]['count'] += 1
        by_type[q_type]['exact_match'] += em
        by_type[q_type]['avg_f1'].append(f1)
        by_type[q_type]['retrieval_recall'] += recall
        
        # By level
        by_level[q_level]['count'] += 1
        by_level[q_level]['exact_match'] += em
        by_level[q_level]['avg_f1'].append(f1)
        by_level[q_level]['retrieval_recall'] += recall
        
        # Track failures
        if not em:
            failed_questions.append({
                'question_id': result.get('question_id'),
                'question': result.get('question', ''),
                'predicted': predicted,
                'gold': gold,
                'f1': f1,
                'retrieval_recall': recall
            })
        
        # Track "Insufficient information" responses
        if 'insufficient information' in predicted.lower():
            insufficient_info_count += 1
    
    # Overall metrics
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)
    
    overall_em = sum(exact_matches) / len(exact_matches) * 100
    overall_f1 = sum(token_f1_scores) / len(token_f1_scores) * 100
    overall_recall = sum(retrieval_recalls) / len(retrieval_recalls) * 100
    
    print(f"\n📊 Answer Quality:")
    print(f"  Exact Match (EM):    {overall_em:.1f}%")
    print(f"  Token F1:            {overall_f1:.1f}%")
    print(f"  Insufficient Info:   {insufficient_info_count} ({insufficient_info_count/len(results)*100:.1f}%)")
    
    print(f"\n📊 Retrieval Quality:")
    print(f"  Retrieval Recall:    {overall_recall:.1f}%")
    
    # By question type
    print("\n" + "=" * 80)
    print("PERFORMANCE BY QUESTION TYPE")
    print("=" * 80)
    
    for q_type, metrics in sorted(by_type.items()):
        count = metrics['count']
        em_pct = metrics['exact_match'] / count * 100
        avg_f1 = sum(metrics['avg_f1']) / count * 100
        recall_pct = metrics['retrieval_recall'] / count * 100
        
        print(f"\n{q_type.upper()} ({count} questions):")
        print(f"  EM:       {em_pct:.1f}%")
        print(f"  F1:       {avg_f1:.1f}%")
        print(f"  Recall:   {recall_pct:.1f}%")
    
    # By question level
    print("\n" + "=" * 80)
    print("PERFORMANCE BY DIFFICULTY LEVEL")
    print("=" * 80)
    
    for q_level, metrics in sorted(by_level.items()):
        count = metrics['count']
        em_pct = metrics['exact_match'] / count * 100
        avg_f1 = sum(metrics['avg_f1']) / count * 100
        recall_pct = metrics['retrieval_recall'] / count * 100
        
        print(f"\n{q_level.upper()} ({count} questions):")
        print(f"  EM:       {em_pct:.1f}%")
        print(f"  F1:       {avg_f1:.1f}%")
        print(f"  Recall:   {recall_pct:.1f}%")
    
    # Failure analysis
    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS")
    print("=" * 80)
    
    # Sort failures by F1 score (lowest first)
    failed_questions.sort(key=lambda x: x['f1'])
    
    print(f"\nTotal Failures: {len(failed_questions)} ({len(failed_questions)/len(results)*100:.1f}%)")
    
    # Show top 10 worst failures
    print("\n🔍 Top 10 Worst Failures (by F1 score):")
    for i, failure in enumerate(failed_questions[:10], 1):
        print(f"\n{i}. Question ID: {failure['question_id']}")
        print(f"   Question: {failure['question'][:80]}...")
        print(f"   Gold:     {failure['gold']}")
        print(f"   Predicted: {failure['predicted']}")
        print(f"   F1: {failure['f1']:.2f}, Retrieval: {'✅' if failure['retrieval_recall'] else '❌'}")
    
    # Retrieval failures
    retrieval_failures = [f for f in failed_questions if not f['retrieval_recall']]
    print(f"\n📉 Retrieval Failures: {len(retrieval_failures)} ({len(retrieval_failures)/len(results)*100:.1f}%)")
    
    # Answer failures (retrieval ok but answer wrong)
    answer_failures = [f for f in failed_questions if f['retrieval_recall']]
    print(f"📉 Answer Failures (retrieval OK): {len(answer_failures)} ({len(answer_failures)/len(results)*100:.1f}%)")
    
    # Save summary
    summary = {
        'overall': {
            'exact_match': overall_em,
            'token_f1': overall_f1,
            'retrieval_recall': overall_recall,
            'insufficient_info_count': insufficient_info_count,
            'total_questions': len(results)
        },
        'by_type': {
            q_type: {
                'count': metrics['count'],
                'exact_match': metrics['exact_match'] / metrics['count'] * 100,
                'avg_f1': sum(metrics['avg_f1']) / metrics['count'] * 100,
                'retrieval_recall': metrics['retrieval_recall'] / metrics['count'] * 100
            }
            for q_type, metrics in by_type.items()
        },
        'by_level': {
            q_level: {
                'count': metrics['count'],
                'exact_match': metrics['exact_match'] / metrics['count'] * 100,
                'avg_f1': sum(metrics['avg_f1']) / metrics['count'] * 100,
                'retrieval_recall': metrics['retrieval_recall'] / metrics['count'] * 100
            }
            for q_level, metrics in by_level.items()
        }
    }
    
    summary_file = 'evaluation_results_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Summary saved to: {summary_file}")
    
    return summary


if __name__ == "__main__":
    results_file = 'multihop_pipeline_200_results.json'
    summary = analyze_results(results_file)
