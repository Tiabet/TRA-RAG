#!/usr/bin/env python3
"""
Test New Multi-hop Pipeline v3 (Original Passages) on 200 Questions
====================================================================
Unified pipeline test script for HotpotQA and MuSiQue datasets.

Usage:
    python test_pipeline_no_link.py                     # Default: HotpotQA
    python test_pipeline_no_link.py --dataset hotpotqa  # HotpotQA
    python test_pipeline_no_link.py --dataset musique   # MuSiQue
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

from new_multihop_pipeline_no_link import NewMultihopPipelineV3
from new_multihop_pipeline_paths_hint import NewMultihopPipelineV11PathsHint
from hybrid_path_retriever import HybridPathRetriever


# Concurrency settings
CONCURRENCY = 100

# Dataset configurations
DATASET_CONFIGS = {
    'hotpotqa': {
        'data_path': 'HotpotQA/hotpotqa_sample_200.json',
        'db_path': 'HotpotQA/metadata_v3.db',
        'bm25_index': 'HotpotQA/bm25_index',
        'embeddings': 'HotpotQA/path_embeddings.npz',
        'result_path': 'Results/test_hotpot_v6_200_results.json',
    },
    'musique': {
        'data_path': 'MuSiQue/musique_sample_200.json',
        'db_path': 'MuSiQue/metadata_v3.db',
        'bm25_index': 'MuSiQue/bm25_index',
        'embeddings': 'MuSiQue/path_embeddings.npz',
        'result_path': 'Results/test_musique_v10_200_results.json',
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
            print(f"[{idx+1:3d}/{total}] [OK] ({elapsed:.1f}s, {num_passages}p) {question[:50]}...")
            print(f"           Gold: {gold_answer}")
            print(f"           Pred: {predicted}")
            
            return {
                'id': qid,
                'question': question,
                'gold_answer': gold_answer,
                'predicted_answer': predicted,
                'answer_aliases': item.get('answer_aliases', []),  # MuSiQue support
                'time': elapsed,
                'num_passages': num_passages,
                'success': True,
                'decomposition': result.get('decomposition')
            }
        else:
            print(f"[{idx+1:3d}/{total}] ✗ ({elapsed:.1f}s) Error: {result.get('error')[:50]}...")
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
    
    # Handle any exceptions that slipped through
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Pipeline v3 on multi-hop QA datasets')
    parser.add_argument('--dataset', type=str, default='hotpotqa',
                        choices=['hotpotqa', 'musique'],
                        help='Dataset to use (default: hotpotqa)')
    parser.add_argument('--pipeline', type=str, default='v11',
                        choices=['v3', 'v11'],
                        help='Pipeline version to run (default: v3). v11 uses paths-as-hints for SQ answering.')
    parser.add_argument('--result_path', type=str, default=None,
                        help='Optional override for output results JSON path')
    parser.add_argument('--max_questions', type=int, default=None,
                        help='Optional limit for number of questions to run (useful for smoke tests)')
    parser.add_argument('--concurrency', type=int, default=CONCURRENCY,
                        help=f'Concurrency level (default: {CONCURRENCY})')
    return parser.parse_args()


async def run_test(args):
    load_dotenv()
    
    # Get dataset config
    dataset = args.dataset.lower()
    config = DATASET_CONFIGS[dataset]
    concurrency = args.concurrency
    pipeline_version = args.pipeline.lower()
    
    print("="*80)
    print(f"Multi-hop Pipeline {pipeline_version.upper()} Test - {dataset.upper()}")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Concurrency: {concurrency}")
    if pipeline_version == 'v11':
        print("Mode: Hybrid Retrieval + Original Passage Answering + Path Hints")
    else:
        print("Mode: Hybrid Retrieval + Original Passage Answering")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize components
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

    if pipeline_version == 'v11':
        pipeline = NewMultihopPipelineV11PathsHint(
            client=client,
            retriever=retriever,
            hotpotqa_path=config['data_path'],
            db_path=config['db_path'],
            top_k_passages=3,
            top_k_paths=10,
            path_fetch_k=50,
            verbose=False,
        )
    else:
        pipeline = NewMultihopPipelineV3(
            client=client,
            retriever=retriever,
            hotpotqa_path=config['data_path'],
            db_path=config['db_path'],
            top_k=3,
            verbose=False
        )
    
    # Load dataset
    print(f"\n📂 Loading data from: {config['data_path']}")
    with open(config['data_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.max_questions is not None:
        data = data[: max(0, args.max_questions)]
    
    total = len(data)
    print(f"   [OK] Loaded {total} questions")

    # Ensure Results directory exists
    Path('Results').mkdir(parents=True, exist_ok=True)

    # Decide output file early so we can write incremental snapshots
    if args.result_path:
        output_file = args.result_path
    else:
        # Default to config path, but make v11 explicit in filename
        if pipeline_version == 'v11':
            if dataset == 'musique':
                output_file = 'Results/test_musique_v11_200_results.json'
            else:
                output_file = 'Results/test_hotpot_v11_200_results.json'
        else:
            output_file = config['result_path']
    
    # Analyze hop distribution (for MuSiQue)
    if dataset == 'musique':
        hop_dist = {}
        for item in data:
            hop = item['_id'].split('hop')[0]
            hop_dist[hop] = hop_dist.get(hop, 0) + 1
        print(f"   Hop distribution: {hop_dist}")
    
    print("-"*80)
    
    # Process in batches
    all_results = []
    start_time = time.time()

    def build_output_snapshot(results_list, is_final: bool):
        elapsed_total = time.time() - start_time
        success_count_local = sum(1 for r in results_list if r.get('success'))
        error_count_local = len(results_list) - success_count_local

        successful_results_local = [r for r in results_list if r.get('success')]
        avg_time_local = (
            sum(r.get('time', 0) for r in successful_results_local) / len(successful_results_local)
            if successful_results_local else 0
        )
        total_passages_local = sum(r.get('num_passages', 0) for r in successful_results_local)
        avg_passages_local = (
            total_passages_local / len(successful_results_local)
            if successful_results_local else 0
        )

        output_local = {
            'config': {
                'pipeline_version': pipeline_version,
                'dataset': dataset,
                'mode': 'hybrid_retrieval_original_passages' if pipeline_version == 'v3' else 'hybrid_retrieval_original_passages_plus_path_hints',
                'concurrency': concurrency,
                'top_k': 3,
                'bm25_weight': 0.4,
                'dense_weight': 0.6
            },
            'summary': {
                'total_questions': total,
                'completed_questions': len(results_list),
                'successful': success_count_local,
                'errors': error_count_local,
                'total_time': elapsed_total,
                'avg_time_per_question': avg_time_local,
                'total_passages': total_passages_local,
                'avg_passages_per_question': avg_passages_local,
                'timestamp': datetime.now().isoformat(),
                'is_final': bool(is_final)
            },
            'results': results_list
        }
        return convert_to_serializable(output_local)

    def write_snapshot(results_list, is_final: bool):
        snapshot = build_output_snapshot(results_list, is_final=is_final)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    try:
        for batch_start in range(0, total, concurrency):
            batch_end = min(batch_start + concurrency, total)
            batch_items = data[batch_start:batch_end]

            print(f"\n>>> Batch {batch_start//concurrency + 1}: Questions {batch_start+1}-{batch_end}")

            batch_results = await run_batch(pipeline, batch_items, batch_start, total)
            all_results.extend(batch_results)

            # Progress summary
            elapsed = time.time() - start_time
            success_count = sum(1 for r in all_results if r['success'])
            error_count = len(all_results) - success_count

            print(f"    Completed: {len(all_results)}/{total} | Success: {success_count} | Errors: {error_count} | Time: {elapsed:.0f}s")

            # Write incremental snapshot so long runs don't lose progress
            write_snapshot(all_results, is_final=False)
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Writing partial results...")
        write_snapshot(all_results, is_final=False)
        pipeline.close()
        raise
    
    total_time = time.time() - start_time
    
    # Final summary
    success_count = sum(1 for r in all_results if r['success'])
    error_count = len(all_results) - success_count
    
    # Calculate statistics
    successful_results = [r for r in all_results if r['success']]
    avg_time = sum(r['time'] for r in successful_results) / len(successful_results) if successful_results else 0
    total_passages = sum(r.get('num_passages', 0) for r in successful_results)
    avg_passages = total_passages / len(successful_results) if successful_results else 0
    
    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)
    print(f"Dataset: {dataset.upper()}")
    if pipeline_version == 'v11':
        print("Pipeline: v11 (Hybrid + Original Passages + Path Hints)")
    else:
        print("Pipeline: v3 (Hybrid Retrieval + Original Passages)")
    print(f"Total Questions: {total}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Average Time per Question: {avg_time:.2f}s")
    print(f"Total Passages Used: {total_passages}")
    print(f"Average Passages per Question: {avg_passages:.1f}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Save final results (overwrite snapshot with final summary)
    write_snapshot(all_results, is_final=True)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nRun evaluation with:")
    print(f"   python evaluate_mrqa.py {output_file}")
    
    pipeline.close()


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_test(args))
