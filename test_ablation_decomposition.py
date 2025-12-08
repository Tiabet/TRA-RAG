#!/usr/bin/env python3
"""
Ablation Study: Query Decomposition Effect
===========================================
Compare pipeline WITH vs WITHOUT query decomposition.

Usage:
    python test_ablation_decomposition.py                     # Default: HotpotQA
    python test_ablation_decomposition.py --dataset hotpotqa
    python test_ablation_decomposition.py --dataset musique
"""

import asyncio
import json
import time
import os
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

from new_multihop_pipeline_v3_no_decomp import NewMultihopPipelineV3NoDecomp
from hybrid_path_retriever import HybridPathRetriever


CONCURRENCY = 100

DATASET_CONFIGS = {
    'hotpotqa': {
        'data_path': 'HotpotQA/hotpotqa_sample_200.json',
        'db_path': 'HotpotQA/metadata_v3.db',
        'bm25_index': 'HotpotQA/bm25_index',
        'embeddings': 'HotpotQA/path_embeddings.npz',
        'result_path': 'Results/ablation_no_decomp_hotpotqa_results.json',
    },
    'musique': {
        'data_path': 'MuSiQue/musique_sample_200.json',
        'db_path': 'MuSiQue/metadata_v3.db',
        'bm25_index': 'MuSiQue/bm25_index',
        'embeddings': 'MuSiQue/path_embeddings.npz',
        'result_path': 'Results/ablation_no_decomp_musique_results.json',
    }
}


async def process_single_question(pipeline, item, idx, total):
    """Process a single question and return result."""
    question = item['question']
    gold_answer = item['answer']
    qid = item.get('_id', f'q{idx}')
    
    start = time.time()
    
    try:
        result = await pipeline.process_question(question)
        elapsed = time.time() - start
        
        if result['success']:
            predicted = result['final_answer']
            num_passages = result.get('num_passages', 0)
            print(f"[{idx+1:3d}/{total}] ✓ ({elapsed:.1f}s, {num_passages}p) {question[:50]}...")
            print(f"           Gold: {gold_answer}")
            print(f"           Pred: {predicted}")
            
            return {
                'id': qid,
                'question': question,
                'gold_answer': gold_answer,
                'predicted_answer': predicted,
                'answer_aliases': item.get('answer_aliases', []),
                'time': elapsed,
                'num_passages': num_passages,
                'success': True,
                'retrieved_passages': result.get('retrieved_passages', [])
            }
        else:
            print(f"[{idx+1:3d}/{total}] ✗ ({elapsed:.1f}s) Error: {result.get('error', '')[:50]}...")
            return {
                'id': qid,
                'question': question,
                'gold_answer': gold_answer,
                'predicted_answer': None,
                'answer_aliases': item.get('answer_aliases', []),
                'time': elapsed,
                'success': False,
                'error': result.get('error')
            }
            
    except Exception as e:
        elapsed = time.time() - start
        print(f"[{idx+1:3d}/{total}] ✗ ({elapsed:.1f}s) Exception: {str(e)[:50]}...")
        return {
            'id': qid,
            'question': question,
            'gold_answer': gold_answer,
            'predicted_answer': None,
            'answer_aliases': item.get('answer_aliases', []),
            'time': elapsed,
            'success': False,
            'error': str(e)
        }


async def run_batch(pipeline, items, start_idx, total):
    """Run a batch of questions concurrently."""
    tasks = []
    for i, item in enumerate(items):
        idx = start_idx + i
        tasks.append(process_single_question(pipeline, item, idx, total))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            idx = start_idx + i
            item = items[i]
            processed_results.append({
                'id': item.get('_id', f'q{idx}'),
                'question': item['question'],
                'gold_answer': item['answer'],
                'predicted_answer': None,
                'answer_aliases': item.get('answer_aliases', []),
                'time': 0,
                'success': False,
                'error': str(r)
            })
        else:
            processed_results.append(r)
    
    return processed_results


def parse_args():
    parser = argparse.ArgumentParser(description='Ablation Study: No Query Decomposition')
    parser.add_argument('--dataset', type=str, default='hotpotqa',
                        choices=['hotpotqa', 'musique'],
                        help='Dataset to use (default: hotpotqa)')
    parser.add_argument('--concurrency', type=int, default=CONCURRENCY,
                        help=f'Concurrency level (default: {CONCURRENCY})')
    return parser.parse_args()


async def main():
    load_dotenv()
    args = parse_args()
    
    dataset = args.dataset.lower()
    config = DATASET_CONFIGS[dataset]
    concurrency = args.concurrency
    
    print("="*80)
    print(f"ABLATION STUDY: No Query Decomposition - {dataset.upper()}")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Mode: Direct Retrieval (NO decomposition)")
    print(f"Concurrency: {concurrency}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6,
        bm25_index_path=config['bm25_index'],
        embeddings_path=config['embeddings']
    )
    
    pipeline = NewMultihopPipelineV3NoDecomp(
        client=client,
        retriever=retriever,
        hotpotqa_path=config['data_path'],
        db_path=config['db_path'],
        top_k=5,
        verbose=False
    )
    
    # Load data
    print(f"\n📂 Loading data from: {config['data_path']}")
    with open(config['data_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    print(f"   ✓ Loaded {total} questions")
    print("-"*80)
    
    # Process
    all_results = []
    start_time = time.time()
    
    for batch_start in range(0, total, concurrency):
        batch_end = min(batch_start + concurrency, total)
        batch_items = data[batch_start:batch_end]
        
        print(f"\n>>> Batch {batch_start//concurrency + 1}: Questions {batch_start+1}-{batch_end}")
        
        batch_results = await run_batch(pipeline, batch_items, batch_start, total)
        all_results.extend(batch_results)
        
        elapsed = time.time() - start_time
        success_count = sum(1 for r in all_results if r['success'])
        print(f"    Completed: {len(all_results)}/{total} | Success: {success_count} | Time: {elapsed:.0f}s")
    
    total_time = time.time() - start_time
    
    # Summary
    success_count = sum(1 for r in all_results if r['success'])
    successful_results = [r for r in all_results if r['success']]
    avg_time = sum(r['time'] for r in successful_results) / len(successful_results) if successful_results else 0
    total_passages = sum(r.get('num_passages', 0) for r in successful_results)
    avg_passages = total_passages / len(successful_results) if successful_results else 0
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETED")
    print("="*80)
    print(f"Dataset: {dataset.upper()}")
    print(f"Mode: NO Query Decomposition")
    print(f"Total Questions: {total}")
    print(f"Successful: {success_count}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Avg Time/Question: {avg_time:.2f}s")
    print(f"Avg Passages/Question: {avg_passages:.1f}")
    print("="*80)
    
    # Save
    output = {
        'config': {
            'experiment': 'ablation_no_decomposition',
            'pipeline_version': 'v3-no-decomp',
            'dataset': dataset,
            'query_decomposition': False,
            'top_k': 5,
            'bm25_weight': 0.4,
            'dense_weight': 0.6
        },
        'summary': {
            'total_questions': total,
            'successful': success_count,
            'total_time': total_time,
            'avg_time_per_question': avg_time,
            'avg_passages_per_question': avg_passages,
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    Path('Results').mkdir(parents=True, exist_ok=True)
    
    with open(config['result_path'], 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {config['result_path']}")
    print(f"\nRun evaluation:")
    print(f"   python evaluate_mrqa.py {config['result_path']}")
    
    pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
