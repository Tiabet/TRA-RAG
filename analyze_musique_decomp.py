#!/usr/bin/env python3
"""
Analyze MuSiQue question decomposition patterns
"""
import json

def analyze_musique_decomposition():
    """Analyze MuSiQue question decomposition"""
    
    with open('MuSiQue/musique_qa_sample_200.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("="*80)
    print("MuSiQue Question Decomposition Analysis")
    print("="*80)
    
    # Analyze decomposition patterns
    decomp_lengths = []
    
    for q in data[:10]:  # Show first 10 examples
        print("\n" + "="*80)
        print(f"Question: {q['question']}")
        print(f"Answer: {q.get('answer', 'N/A')}")
        
        decomp = q.get('question_decomposition', [])
        decomp_lengths.append(len(decomp))
        
        print(f"\nDecomposition ({len(decomp)} steps):")
        for i, step in enumerate(decomp, 1):
            print(f"\n  Step {i}:")
            print(f"    ID: {step.get('id', 'N/A')}")
            print(f"    Question: {step.get('question', 'N/A')}")
            print(f"    Answer: {step.get('answer', 'N/A')}")
            
            # Check for dependencies
            if 'depends_on' in step:
                print(f"    Depends on: {step['depends_on']}")
    
    # Statistics
    print("\n" + "="*80)
    print("Statistics")
    print("="*80)
    print(f"Total questions analyzed: {len(data)}")
    print(f"Sample size shown: 10")
    
    all_decomp_lengths = []
    for q in data:
        decomp = q.get('question_decomposition', [])
        all_decomp_lengths.append(len(decomp))
    
    from collections import Counter
    length_dist = Counter(all_decomp_lengths)
    
    print("\nDecomposition Length Distribution (all 200 questions):")
    for length, count in sorted(length_dist.items()):
        print(f"  {length} steps: {count} questions ({count/len(data)*100:.1f}%)")
    
    print(f"\nAverage decomposition length: {sum(all_decomp_lengths)/len(all_decomp_lengths):.2f}")
    print(f"Min: {min(all_decomp_lengths)}, Max: {max(all_decomp_lengths)}")


if __name__ == "__main__":
    analyze_musique_decomposition()
