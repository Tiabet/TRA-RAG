#!/usr/bin/env python3
"""
Analyze question types across different datasets
"""
import json
from collections import Counter
from typing import Dict, List

def analyze_dataset(file_path: str, dataset_name: str):
    """Analyze question types and characteristics"""
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print("="*80)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nTotal questions: {len(data)}")
    
    # Analyze question types
    types = Counter()
    for q in data:
        qtype = q.get('type', 'unknown')
        types[qtype] += 1
    
    print("\nQuestion Type Distribution:")
    for qtype, count in types.most_common():
        percentage = count / len(data) * 100
        print(f"  {qtype:20s}: {count:3d} ({percentage:5.1f}%)")
    
    # Show sample questions for each type
    print("\n" + "="*80)
    print("Sample Questions by Type")
    print("="*80)
    
    seen_types = set()
    samples_per_type = {}
    
    for q in data:
        qtype = q.get('type', 'unknown')
        if qtype not in samples_per_type:
            samples_per_type[qtype] = []
        if len(samples_per_type[qtype]) < 3:  # 3 samples per type
            samples_per_type[qtype].append(q)
    
    for qtype in types.keys():
        print(f"\n{'='*80}")
        print(f"Type: {qtype.upper()}")
        print('='*80)
        
        samples = samples_per_type.get(qtype, [])
        for i, q in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(f"  ID: {q.get('_id', 'N/A')}")
            print(f"  Question: {q['question']}")
            print(f"  Answer: {q.get('answer', 'N/A')}")
            
            # Show additional fields if available
            if 'question_decomposition' in q:
                print(f"  Has decomposition: Yes")
            if 'supporting_facts' in q:
                print(f"  Supporting facts: {len(q['supporting_facts'])}")
            if 'evidences' in q:
                print(f"  Evidences: {len(q['evidences'])}")
    
    print("\n" + "="*80)
    print(f"Analysis complete for {dataset_name}")
    print("="*80)
    
    return {
        'total': len(data),
        'types': dict(types),
        'samples': samples_per_type
    }


def main():
    print("\n" + "="*80)
    print("MULTI-HOP QA DATASET ANALYSIS")
    print("="*80)
    
    datasets = [
        ('HotpotQA/hotpotqa_sample_200.json', 'HotpotQA'),
        ('2WikiMultihopQA/2wikimultihopqa_sample_200.json', '2WikiMultihopQA'),
        ('MuSiQue/musique_qa_sample_200.json', 'MuSiQue')
    ]
    
    results = {}
    
    for file_path, dataset_name in datasets:
        try:
            results[dataset_name] = analyze_dataset(file_path, dataset_name)
        except FileNotFoundError:
            print(f"\n❌ File not found: {file_path}")
        except Exception as e:
            print(f"\n❌ Error analyzing {dataset_name}: {e}")
        
        print("\n\n")
    
    # Summary comparison
    print("\n" + "="*80)
    print("CROSS-DATASET COMPARISON")
    print("="*80)
    
    all_types = set()
    for dataset_name, result in results.items():
        all_types.update(result['types'].keys())
    
    print("\nQuestion Type Coverage:")
    print(f"{'Type':<20s} | {'HotpotQA':>12s} | {'2WikiMultihop':>12s} | {'MuSiQue':>12s}")
    print("-" * 80)
    
    for qtype in sorted(all_types):
        row = f"{qtype:<20s}"
        for dataset_name in ['HotpotQA', '2WikiMultihopQA', 'MuSiQue']:
            if dataset_name in results:
                count = results[dataset_name]['types'].get(qtype, 0)
                row += f" | {count:12d}"
            else:
                row += f" | {'N/A':>12s}"
        print(row)
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
