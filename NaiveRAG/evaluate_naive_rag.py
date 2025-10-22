"""
범용 NaiveRAG 평가 스크립트 (병렬 처리)
=========================================
다양한 QA 데이터셋에 대한 retrieval 성능 평가
- 병렬 처리로 속도 대폭 개선
- 데이터셋별 설정 지원
"""

import json
import os
import sys
from typing import Dict, List, Tuple
from tqdm import tqdm
from naive_rag import NaiveRAG
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_single_query(rag: NaiveRAG, data: Dict, top_k: int) -> Dict:
    """단일 쿼리 처리 (병렬 실행용)"""
    query = data['question']
    doc_id = data['_id']
    question_type = data.get('type', 'unknown')
    level = data.get('level', 'unknown')
    
    try:
        # Retrieval
        retrieved = rag.retrieve(query, top_k=top_k)
        
        # Retrieval 평가
        eval_result = rag.evaluate_retrieval(doc_id, retrieved)
        
        # 검색된 passage titles 저장
        retrieved_titles = [chunk.title for chunk, score in retrieved]
        retrieved_scores = [float(score) for chunk, score in retrieved]
        
        return {
            'success': True,
            'question_id': doc_id,
            'question': query,
            'question_type': question_type,
            'level': level,
            'retrieved_titles': retrieved_titles,
            'retrieved_scores': retrieved_scores,
            'retrieval_recall': eval_result['recall'],
            'retrieval_precision': eval_result['precision'],
            'retrieval_f1': eval_result['f1'],
            'gold_count': eval_result['gold_count'],
            'correct_count': eval_result.get('correct_count', 0)
        }
    except Exception as e:
        return {
            'success': False,
            'question_id': doc_id,
            'error': str(e)
        }


def evaluate_full_dataset(rag: NaiveRAG, data_path: str, top_k: int = 5, 
                         limit: int = None, max_workers: int = 8):
    """전체 데이터셋 평가 (병렬 처리)
    
    Args:
        rag: NaiveRAG 인스턴스
        data_path: 데이터 파일 경로
        top_k: 검색할 passage 개수
        limit: 평가할 질문 개수 제한 (None이면 전체)
        max_workers: 병렬 처리 워커 수 (기본값: 8)
    """
    
    retrieval_metrics = {
        'recall': [],
        'precision': [],
        'f1': []
    }
    
    results = []
    errors = []
    
    print(f"\n{'='*60}")
    print(f"Evaluating NaiveRAG Retrieval (Parallel)")
    print(f"Data: {os.path.basename(data_path)}")
    print(f"Top-K: {top_k}")
    print(f"Max Workers: {max_workers}")
    print(f"{'='*60}\n")
    
    # 데이터 로드
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if limit:
            lines = lines[:limit]
    
    data_list = [json.loads(line) for line in lines]
    
    # 병렬 처리
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 작업 제출
        future_to_data = {
            executor.submit(process_single_query, rag, data, top_k): data 
            for data in data_list
        }
        
        # Progress bar와 함께 결과 수집
        for future in tqdm(as_completed(future_to_data), total=len(data_list), 
                          desc="Evaluating Retrieval"):
            result = future.result()
            
            if result['success']:
                retrieval_metrics['recall'].append(result['retrieval_recall'])
                retrieval_metrics['precision'].append(result['retrieval_precision'])
                retrieval_metrics['f1'].append(result['retrieval_f1'])
                results.append(result)
            else:
                errors.append(result)
    
    if errors:
        print(f"\n⚠️  {len(errors)} errors occurred during evaluation")
        for err in errors[:5]:  # 처음 5개만 출력
            print(f"  - Question {err['question_id']}: {err['error']}")
    
    # 전체 통계 계산
    print(f"\n{'='*60}")
    print("Overall Retrieval Performance")
    print(f"{'='*60}")
    avg_recall = np.mean(retrieval_metrics['recall'])
    avg_precision = np.mean(retrieval_metrics['precision'])
    avg_f1 = np.mean(retrieval_metrics['f1'])
    
    print(f"Total Questions:   {len(results)}")
    print(f"Average Recall:    {avg_recall:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average F1:        {avg_f1:.4f}")
    
    # 질문 타입별 분석
    print(f"\n{'='*60}")
    print("Performance by Question Type")
    print(f"{'='*60}")
    
    type_stats = defaultdict(lambda: {'recall': [], 'precision': [], 'f1': []})
    for result in results:
        qtype = result['question_type']
        type_stats[qtype]['recall'].append(result['retrieval_recall'])
        type_stats[qtype]['precision'].append(result['retrieval_precision'])
        type_stats[qtype]['f1'].append(result['retrieval_f1'])
    
    for qtype, stats in sorted(type_stats.items()):
        print(f"\n{qtype}:")
        print(f"  Count:      {len(stats['recall'])}")
        print(f"  Recall:     {np.mean(stats['recall']):.4f}")
        print(f"  Precision:  {np.mean(stats['precision']):.4f}")
        print(f"  F1:         {np.mean(stats['f1']):.4f}")
    
    # 난이도별 분석
    print(f"\n{'='*60}")
    print("Performance by Difficulty Level")
    print(f"{'='*60}")
    
    level_stats = defaultdict(lambda: {'recall': [], 'precision': [], 'f1': []})
    for result in results:
        level = result['level']
        level_stats[level]['recall'].append(result['retrieval_recall'])
        level_stats[level]['precision'].append(result['retrieval_precision'])
        level_stats[level]['f1'].append(result['retrieval_f1'])
    
    for level, stats in sorted(level_stats.items()):
        print(f"\n{level}:")
        print(f"  Count:      {len(stats['recall'])}")
        print(f"  Recall:     {np.mean(stats['recall']):.4f}")
        print(f"  Precision:  {np.mean(stats['precision']):.4f}")
        print(f"  F1:         {np.mean(stats['f1']):.4f}")
    
    return {
        'retrieval_metrics': retrieval_metrics,
        'results': results,
        'overall': {
            'total_questions': len(results),
            'avg_recall': avg_recall,
            'avg_precision': avg_precision,
            'avg_retrieval_f1': avg_f1
        }
    }


def save_results(results: Dict, output_path: str):
    """결과 저장"""
    # numpy 타입을 파이썬 기본 타입으로 변환
    def convert_to_serializable(obj):
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_path}")


def main():
    """메인 실행 - 명령행 인자 지원"""
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_naive_rag.py <data_path> [index_path] [results_path] [options]")
        print("\nExamples:")
        print("  python evaluate_naive_rag.py ../HotpotQA/hotpot.jsonl")
        print("  python evaluate_naive_rag.py ../2WikiMultihopQA/2wiki.jsonl --top-k 10 --limit 200")
        print("\nOptions:")
        print("  --top-k N         Number of passages to retrieve (default: 5)")
        print("  --limit N         Limit number of questions to evaluate")
        print("  --max-workers N   Number of parallel workers (default: 8)")
        sys.exit(1)
    
    DATA_PATH = sys.argv[1]
    
    # 기본 경로 설정
    base_dir = os.path.dirname(DATA_PATH)
    base_name = os.path.basename(DATA_PATH).replace('.jsonl', '')
    
    INDEX_PATH = os.path.join(base_dir, f"{base_name}_naive_rag_index.json")
    RESULTS_PATH = os.path.join(base_dir, f"{base_name}_naive_rag_results.json")
    
    # 명령행 인자에서 경로 오버라이드
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        INDEX_PATH = sys.argv[2]
    if len(sys.argv) > 3 and not sys.argv[3].startswith('--'):
        RESULTS_PATH = sys.argv[3]
    
    # 옵션 파싱
    top_k = 5
    limit = None
    max_workers = 8
    
    for i, arg in enumerate(sys.argv):
        if arg == '--top-k' and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])
        elif arg == '--limit' and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])
        elif arg == '--max-workers' and i + 1 < len(sys.argv):
            max_workers = int(sys.argv[i + 1])
    
    print(f"Configuration:")
    print(f"  Data: {DATA_PATH}")
    print(f"  Index: {INDEX_PATH}")
    print(f"  Results: {RESULTS_PATH}")
    print(f"  Top-K: {top_k}")
    print(f"  Limit: {limit if limit else 'None (all data)'}")
    print(f"  Max Workers: {max_workers}")
    
    # NaiveRAG 로드
    rag = NaiveRAG()
    
    if not os.path.exists(INDEX_PATH):
        print(f"\nIndex not found at {INDEX_PATH}")
        print("Creating index first...")
        rag.load_data(DATA_PATH)
        rag.create_embeddings(batch_size=100)
        rag.save_index(INDEX_PATH)
    else:
        rag.load_index(INDEX_PATH)
    
    # 전체 평가 (병렬 처리)
    results = evaluate_full_dataset(
        rag, 
        DATA_PATH, 
        top_k=top_k,
        limit=limit,
        max_workers=max_workers
    )
    
    # 결과 저장
    save_results(results, RESULTS_PATH)


if __name__ == "__main__":
    main()
