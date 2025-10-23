#!/usr/bin/env python
"""
Evaluate Retrieval Recall for Multi-hop QA
============================================
Measures how many gold supporting facts are successfully retrieved.

Metrics:
- Recall: retrieved_gold / total_unique_gold (per query, then averaged)
- Perfect Recall: percentage of queries where all gold facts are retrieved
"""
import json
from pathlib import Path
from collections import defaultdict


def evaluate_retrieval_recall(results_path: Path, verbose: bool = False):
    """
    Evaluate retrieval recall from multihop pipeline results.
    
    Args:
        results_path: Path to multihop_pipeline_200_results.json
        verbose: If True, print per-query details
        
    Returns:
        dict with overall metrics
    """
    with results_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    total_queries = 0
    total_gold_facts = 0
    total_retrieved_gold = 0
    perfect_recall_count = 0
    
    recall_scores = []
    per_type_stats = defaultdict(lambda: {'queries': 0, 'gold': 0, 'retrieved': 0, 'perfect': 0})
    
    failed_queries = []  # Track queries with recall < 1.0
    
    for item in results:
        question = item.get('question', 'N/A')
        question_type = item.get('question_type', 'unknown')
        gold_facts_raw = item.get('gold_supporting_facts', [])
        retrieved_titles = item.get('retrieved_passages', {}).get('titles', [])
        
        # Extract unique gold titles (deduplication)
        # gold_supporting_facts: [[title, sent_id], ...]
        gold_titles = set()
        for fact in gold_facts_raw:
            if isinstance(fact, list) and len(fact) >= 1:
                gold_titles.add(fact[0])
        
        # Retrieved titles (already unique in most cases, but ensure set)
        retrieved_set = set(retrieved_titles)
        
        # Calculate overlap
        retrieved_gold = gold_titles & retrieved_set
        
        num_gold = len(gold_titles)
        num_retrieved_gold = len(retrieved_gold)
        
        # Query-level recall
        if num_gold > 0:
            query_recall = num_retrieved_gold / num_gold
        else:
            query_recall = 0.0  # No gold facts (edge case)
        
        recall_scores.append(query_recall)
        
        # Perfect recall check
        if query_recall == 1.0:
            perfect_recall_count += 1
        else:
            failed_queries.append({
                'question': question[:100],
                'gold_titles': list(gold_titles),
                'retrieved_titles': list(retrieved_set),
                'missing': list(gold_titles - retrieved_set),
                'recall': query_recall
            })
        
        # Aggregate stats
        total_queries += 1
        total_gold_facts += num_gold
        total_retrieved_gold += num_retrieved_gold
        
        # Per-type stats
        per_type_stats[question_type]['queries'] += 1
        per_type_stats[question_type]['gold'] += num_gold
        per_type_stats[question_type]['retrieved'] += num_retrieved_gold
        if query_recall == 1.0:
            per_type_stats[question_type]['perfect'] += 1
        
        # Verbose output
        if verbose:
            missing = gold_titles - retrieved_set
            status = "✅" if query_recall == 1.0 else "❌"
            print(f"{status} Recall: {query_recall:.3f} | Gold: {num_gold} | Retrieved: {num_retrieved_gold}")
            print(f"   Q: {question[:80]}...")
            if missing:
                print(f"   Missing: {missing}")
            print()
    
    # Overall metrics
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    perfect_recall_rate = perfect_recall_count / total_queries if total_queries > 0 else 0.0
    
    # Macro recall (average per query)
    macro_recall = avg_recall
    
    # Micro recall (total retrieved / total gold across all queries)
    micro_recall = total_retrieved_gold / total_gold_facts if total_gold_facts > 0 else 0.0
    
    metrics = {
        'total_queries': total_queries,
        'total_gold_facts': total_gold_facts,
        'total_retrieved_gold': total_retrieved_gold,
        'macro_recall': macro_recall,  # Average of per-query recalls
        'micro_recall': micro_recall,  # Overall retrieved/gold ratio
        'perfect_recall_count': perfect_recall_count,
        'perfect_recall_rate': perfect_recall_rate,
        'by_type': {}
    }
    
    # Per-type metrics
    for qtype, stats in per_type_stats.items():
        if stats['queries'] > 0:
            type_recall = stats['retrieved'] / stats['gold'] if stats['gold'] > 0 else 0.0
            type_perfect_rate = stats['perfect'] / stats['queries']
            metrics['by_type'][qtype] = {
                'queries': stats['queries'],
                'gold_facts': stats['gold'],
                'retrieved_gold': stats['retrieved'],
                'recall': type_recall,
                'perfect_recall_rate': type_perfect_rate
            }
    
    return metrics, failed_queries


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate retrieval recall from multihop results")
    parser.add_argument('--results', type=Path, default=Path('multihop_pipeline_200_results.json'),
                        help='Path to results JSON file')
    parser.add_argument('--verbose', action='store_true', help='Print per-query details')
    parser.add_argument('--out', type=Path, default=None, help='Optional output JSON for metrics')
    parser.add_argument('--show-failures', action='store_true', help='Show queries with recall < 1.0')
    parser.add_argument('--top-n-failures', type=int, default=10, help='Number of failures to show')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Retrieval Recall Evaluation")
    print("=" * 80)
    print(f"Results file: {args.results}")
    print()
    
    metrics, failed_queries = evaluate_retrieval_recall(args.results, verbose=args.verbose)
    
    # Print summary
    print("=" * 80)
    print("Overall Metrics")
    print("=" * 80)
    print(f"Total Queries       : {metrics['total_queries']}")
    print(f"Total Gold Facts    : {metrics['total_gold_facts']}")
    print(f"Retrieved Gold Facts: {metrics['total_retrieved_gold']}")
    print()
    print(f"Macro Recall (avg per query): {metrics['macro_recall']:.3f}")
    print(f"Micro Recall (total ratio)  : {metrics['micro_recall']:.3f}")
    print()
    print(f"Perfect Recall Rate: {metrics['perfect_recall_rate']:.3f} ({metrics['perfect_recall_count']}/{metrics['total_queries']})")
    print()
    
    # Per-type breakdown
    if metrics['by_type']:
        print("=" * 80)
        print("By Question Type")
        print("=" * 80)
        for qtype, stats in metrics['by_type'].items():
            print(f"{qtype.upper()}:")
            print(f"  Queries        : {stats['queries']}")
            print(f"  Gold Facts     : {stats['gold_facts']}")
            print(f"  Retrieved Gold : {stats['retrieved_gold']}")
            print(f"  Recall         : {stats['recall']:.3f}")
            print(f"  Perfect Recall : {stats['perfect_recall_rate']:.3f} ({int(stats['perfect_recall_rate'] * stats['queries'])}/{stats['queries']})")
            print()
    
    # Show failures
    if args.show_failures and failed_queries:
        print("=" * 80)
        print(f"Top {args.top_n_failures} Queries with Incomplete Retrieval")
        print("=" * 80)
        for i, fail in enumerate(failed_queries[:args.top_n_failures], 1):
            print(f"{i}. Recall: {fail['recall']:.3f}")
            print(f"   Q: {fail['question']}...")
            print(f"   Missing: {fail['missing']}")
            print()
    
    # Save to JSON
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Metrics saved to: {args.out}")


if __name__ == "__main__":
    main()
