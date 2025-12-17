import json
import argparse
import numpy as np
from typing import Set, List, Dict

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict) and 'results' in data:
            return data['results']
        return data

def get_gold_titles(item) -> Set[str]:
    """Extract unique titles from supporting_facts."""
    return set(fact[0] for fact in item.get('supporting_facts', []))

def get_retrieved_titles(result_item) -> Set[str]:
    """Extract unique titles from retrieved passages in the result."""
    titles = set()
    
    # 1. Check top-level retrieved_passages (for No-QD pipelines)
    if 'retrieved_passages' in result_item:
        passages = result_item['retrieved_passages']
        for p in passages:
            if isinstance(p, dict):
                titles.add(p.get('title', ''))
            elif isinstance(p, str):
                titles.add(p)
    
    # 1.1 Check retrieved_docs (simple list of titles)
    if 'retrieved_docs' in result_item:
        for t in result_item['retrieved_docs']:
            if isinstance(t, str):
                titles.add(t)

    # 2. Check decomposition for retrieved passages (for QD pipelines)
    decomposition = result_item.get('decomposition')
    if decomposition:
        subquestions = []
        if isinstance(decomposition, list):
            subquestions = decomposition
        elif isinstance(decomposition, dict):
            subquestions = decomposition.get('subquestions', [])
            
        for sq in subquestions:
            passages = sq.get('retrieved_passages', [])
            for p in passages:
                if isinstance(p, dict):
                    titles.add(p.get('title', ''))
                elif isinstance(p, str):
                    titles.add(p)
                    
    return titles

def evaluate(result_path, gold_path):
    results = load_json(result_path)
    gold_data = load_json(gold_path)
    
    # Create a map for gold data
    gold_map = {item['_id']: item for item in gold_data}
    # Also map by question text as fallback if ID is missing or different
    gold_q_map = {item['question']: item for item in gold_data}
    
    metrics = {
        'recall': [],
        'precision': [],
        'f1': [],
        'hit_rate': [] # 1 if recall == 1.0 else 0 (All gold passages retrieved)
    }
    
    print(f"Evaluating {len(results)} results...")
    
    count = 0
    for res in results:
        question = res['question']
        
        # Find corresponding gold item
        gold_item = gold_q_map.get(question)
        if not gold_item:
            # Try finding by ID if available
            if '_id' in res and res['_id'] in gold_map:
                gold_item = gold_map[res['_id']]
        
        if not gold_item:
            # print(f"Warning: Gold item not found for question: {question[:50]}...")
            continue
            
        gold_titles = get_gold_titles(gold_item)
        retrieved_titles = get_retrieved_titles(res)
        
        # Calculate metrics
        intersection = gold_titles.intersection(retrieved_titles)
        
        recall = len(intersection) / len(gold_titles) if gold_titles else 0
        precision = len(intersection) / len(retrieved_titles) if retrieved_titles else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['recall'].append(recall)
        metrics['precision'].append(precision)
        metrics['f1'].append(f1)
        metrics['hit_rate'].append(1.0 if recall == 1.0 else 0.0)
        
        count += 1
        
    print(f"\nEvaluation Results ({count} questions):")
    print(f"Avg Recall:    {np.mean(metrics['recall']):.4f}")
    print(f"Avg Precision: {np.mean(metrics['precision']):.4f}")
    print(f"Avg F1:        {np.mean(metrics['f1']):.4f}")
    print(f"Hit Rate (All):{np.mean(metrics['hit_rate']):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default ='Results/test_hotpot_v11_200_results.json', help='Path to result JSON file')
    # parser.add_argument('--gold_path', type=str, default='MuSiQue/musique_sample_200.json', help='Path to gold dataset')
    parser.add_argument('--gold_path', type=str, default='HotpotQA/hotpotqa_sample_200.json', help='Path to gold dataset')
    args = parser.parse_args()
    
    evaluate(args.result_path, args.gold_path)