"""
Analyze Perfect Recall but Low Accuracy Cases
==============================================
Perfect Recall = 모든 supporting facts를 찾았음 (Recall = 1.0)
Low Accuracy = 정답 문자열이 예측에 포함되지 않음

이런 경우들을 분석하여 원인을 파악합니다.
"""

import json
import re
import string
from collections import Counter
from pathlib import Path

def normalize(s: str) -> str:
    """Text normalization for comparison"""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())

def load_results():
    """Load evaluation results"""
    results_path = Path('multihop_pipeline_200_results.json')
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with results_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['results']

def load_gold_data():
    """Load gold supporting facts"""
    gold_path = Path('HotpotQA/hotpotqa_sample_200.json')
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")
    
    with gold_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Build mapping: question_id -> supporting_facts
    gold_map = {}
    for item in data:
        qid = item.get('_id')
        supporting_facts = item.get('supporting_facts', [])
        gold_map[qid] = {
            'question': item.get('question'),
            'answer': item.get('answer'),
            'supporting_facts': supporting_facts
        }
    
    return gold_map

def calculate_recall(retrieved_titles, supporting_facts):
    """
    Calculate retrieval recall.
    supporting_facts = [["Title1", 0], ["Title2", 1], ...]
    """
    if not supporting_facts:
        return 0.0
    
    # Extract unique titles from supporting facts
    gold_titles = set(title for title, _ in supporting_facts)
    
    # Normalize titles for comparison
    gold_titles_norm = {normalize(t) for t in gold_titles}
    retrieved_titles_norm = {normalize(t) for t in retrieved_titles}
    
    # Count matches
    matched = gold_titles_norm & retrieved_titles_norm
    
    recall = len(matched) / len(gold_titles) if gold_titles else 0.0
    
    return recall, matched, gold_titles_norm

def check_accuracy(predicted_answer, gold_answer):
    """Check if gold answer is contained in prediction"""
    pred_norm = normalize(predicted_answer)
    gold_norm = normalize(gold_answer)
    
    return gold_norm in pred_norm

def main():
    print("="*100)
    print("Analyzing Perfect Recall but Low Accuracy Cases")
    print("="*100)
    
    # Load data
    results = load_results()
    gold_data = load_gold_data()
    
    # Analyze each result
    perfect_recall_cases = []
    low_accuracy_cases = []
    perfect_recall_low_accuracy = []
    
    for result in results:
        qid = result['question_id']
        
        if qid not in gold_data:
            continue
        
        gold = gold_data[qid]
        supporting_facts = gold['supporting_facts']
        
        # Get retrieved titles
        retrieved_titles = []
        if 'retrieved_passages' in result:
            retrieved_titles = result['retrieved_passages'].get('titles', [])
        
        # Calculate recall
        recall, matched, gold_titles = calculate_recall(retrieved_titles, supporting_facts)
        
        # Check accuracy
        predicted = result.get('predicted_answer', '')
        gold_answer = result.get('gold_answer', '')
        is_accurate = check_accuracy(predicted, gold_answer)
        
        # Categorize
        if recall == 1.0:
            perfect_recall_cases.append({
                'qid': qid,
                'question': result['question'],
                'gold_answer': gold_answer,
                'predicted_answer': predicted,
                'is_accurate': is_accurate,
                'retrieved_titles': retrieved_titles,
                'gold_titles': list(gold_titles),
                'matched_titles': list(matched)
            })
            
            if not is_accurate:
                low_accuracy_cases.append(perfect_recall_cases[-1])
                perfect_recall_low_accuracy.append(perfect_recall_cases[-1])
    
    # Print summary
    print(f"\nTotal Questions: {len(results)}")
    print(f"Perfect Recall Cases: {len(perfect_recall_cases)}/{len(results)} ({len(perfect_recall_cases)/len(results)*100:.1f}%)")
    
    if len(perfect_recall_cases) > 0:
        print(f"Perfect Recall + Low Accuracy: {len(perfect_recall_low_accuracy)}/{len(perfect_recall_cases)} ({len(perfect_recall_low_accuracy)/len(perfect_recall_cases)*100:.1f}%)")
    else:
        print(f"Perfect Recall + Low Accuracy: No perfect recall cases found")
        print("\nNote: Check if supporting facts are being properly retrieved.")
        return
    
    print(f"\n{'='*100}")
    print(f"Perfect Recall but Low Accuracy Cases ({len(perfect_recall_low_accuracy)} cases)")
    print(f"{'='*100}")
    
    # Show detailed examples
    for i, case in enumerate(perfect_recall_low_accuracy[:10], 1):  # First 10 cases
        print(f"\n{'─'*100}")
        print(f"Case {i}/{len(perfect_recall_low_accuracy)}")
        print(f"{'─'*100}")
        print(f"Question: {case['question']}")
        print(f"\nGold Answer: {case['gold_answer']}")
        print(f"Predicted Answer: {case['predicted_answer']}")
        print(f"\nGold Titles (all retrieved ✓):")
        for title in case['gold_titles']:
            print(f"  - {title}")
        print(f"\nTotal Retrieved Titles: {len(case['retrieved_titles'])}")
    
    # Save detailed analysis
    output_path = Path('perfect_recall_low_accuracy_analysis.json')
    with output_path.open('w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_questions': len(results),
                'perfect_recall_count': len(perfect_recall_cases),
                'perfect_recall_rate': len(perfect_recall_cases) / len(results),
                'low_accuracy_in_perfect_recall': len(perfect_recall_low_accuracy),
                'low_accuracy_rate': len(perfect_recall_low_accuracy) / len(perfect_recall_cases) if perfect_recall_cases else 0
            },
            'cases': perfect_recall_low_accuracy
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*100}")
    print(f"✅ Detailed analysis saved to: {output_path}")
    print(f"{'='*100}")
    
    # Analyze common patterns
    print(f"\n{'='*100}")
    print("Common Patterns in Low Accuracy Cases")
    print(f"{'='*100}")
    
    # Pattern 1: Answer format mismatch
    format_mismatches = []
    for case in perfect_recall_low_accuracy:
        gold = normalize(case['gold_answer'])
        pred = normalize(case['predicted_answer'])
        
        # Check if words are present but format is different
        gold_words = set(gold.split())
        pred_words = set(pred.split())
        
        common_words = gold_words & pred_words
        if len(common_words) > 0 and len(common_words) >= len(gold_words) * 0.5:
            format_mismatches.append(case)
    
    print(f"\n1. Format Mismatch (words present but format different): {len(format_mismatches)}/{len(perfect_recall_low_accuracy)}")
    if format_mismatches:
        example = format_mismatches[0]
        print(f"   Example:")
        print(f"   Gold: {example['gold_answer']}")
        print(f"   Pred: {example['predicted_answer']}")
    
    # Pattern 2: Completely wrong answer
    wrong_answers = []
    for case in perfect_recall_low_accuracy:
        gold = normalize(case['gold_answer'])
        pred = normalize(case['predicted_answer'])
        
        gold_words = set(gold.split())
        pred_words = set(pred.split())
        
        common_words = gold_words & pred_words
        if len(common_words) == 0 or len(common_words) < len(gold_words) * 0.3:
            wrong_answers.append(case)
    
    print(f"\n2. Completely Wrong Answer (< 30% word overlap): {len(wrong_answers)}/{len(perfect_recall_low_accuracy)}")
    if wrong_answers:
        example = wrong_answers[0]
        print(f"   Example:")
        print(f"   Question: {example['question']}")
        print(f"   Gold: {example['gold_answer']}")
        print(f"   Pred: {example['predicted_answer']}")

if __name__ == "__main__":
    main()
