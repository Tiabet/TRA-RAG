"""
NaiveRAG 평가 결과 요약 출력
=============================
다양한 데이터셋의 결과 파일을 읽어서 요약 출력
"""

import json
import sys
import os


def show_results(results_path: str):
    """결과 파일을 읽어서 요약 출력"""
    
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        print("Please run evaluate_naive_rag.py first.")
        return
    
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    overall = data['overall']
    
    print("\n" + "="*60)
    print("NaiveRAG Retrieval Performance Summary")
    print("="*60)
    print(f"\nDataset: {os.path.basename(results_path)}")
    print(f"Total Questions: {overall['total_questions']}")
    print(f"\nAverage Metrics:")
    print(f"  Recall:    {overall['avg_recall']:.4f}")
    print(f"  Precision: {overall['avg_precision']:.4f}")
    print(f"  F1 Score:  {overall['avg_retrieval_f1']:.4f}")
    
    # Top 5 best performing questions
    results = data['results']
    sorted_by_f1 = sorted(results, key=lambda x: x['retrieval_f1'], reverse=True)
    
    print(f"\n{'='*60}")
    print("Top 5 Best Retrieval Performance")
    print("="*60)
    for i, r in enumerate(sorted_by_f1[:5], 1):
        print(f"\n{i}. F1: {r['retrieval_f1']:.4f} | Type: {r['question_type']} | Level: {r['level']}")
        print(f"   Q: {r['question'][:80]}...")
        if r.get('retrieved_titles'):
            print(f"   Retrieved: {', '.join(r['retrieved_titles'][:3])}")
    
    # Bottom 5 worst performing questions
    print(f"\n{'='*60}")
    print("Top 5 Worst Retrieval Performance")
    print("="*60)
    for i, r in enumerate(sorted_by_f1[-5:], 1):
        print(f"\n{i}. F1: {r['retrieval_f1']:.4f} | Type: {r['question_type']} | Level: {r['level']}")
        print(f"   Q: {r['question'][:80]}...")
        if r.get('retrieved_titles'):
            print(f"   Retrieved: {', '.join(r['retrieved_titles'][:3])}")
    
    # 질문 타입별 요약
    type_counts = {}
    for r in results:
        qtype = r.get('question_type', 'unknown')
        if qtype not in type_counts:
            type_counts[qtype] = 0
        type_counts[qtype] += 1
    
    if type_counts:
        print(f"\n{'='*60}")
        print("Questions by Type")
        print("="*60)
        for qtype, count in sorted(type_counts.items()):
            print(f"  {qtype}: {count}")
    
    print("\n" + "="*60 + "\n")


def main():
    """메인 실행"""
    if len(sys.argv) < 2:
        print("Usage: python show_results.py <results_path>")
        print("\nExamples:")
        print("  python show_results.py ../HotpotQA/hotpot_naive_rag_results.json")
        print("  python show_results.py ../2WikiMultihopQA/2wiki_naive_rag_results.json")
        sys.exit(1)
    
    results_path = sys.argv[1]
    
    try:
        show_results(results_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
