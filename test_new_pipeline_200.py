#!/usr/bin/env python3
"""
Test New Multi-hop Pipeline on 200 HotpotQA Questions
======================================================
Runs pipeline with high concurrency for speed.
Results are saved for separate evaluation.
"""

import asyncio
import json
import time
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

from new_multihop_pipeline import NewMultihopPipeline
from hybrid_path_retriever import HybridPathRetriever


# Concurrency settings
CONCURRENCY = 100  # Number of parallel questions


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
                'time': 0,
                'success': False,
                'error': str(r)
            })
        else:
            processed_results.append(r)
    
    return processed_results


async def run_test():
    load_dotenv()
    
    print("="*80)
    print("New Multi-hop Pipeline - 200 Questions Test")
    print("="*80)
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize components
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6
    )
    
    pipeline = NewMultihopPipeline(
        client=client,
        retriever=retriever,
        top_k=3,
        verbose=False
    )
    
    # Load HotpotQA sample
    with open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8') as f:
        hotpot_data = json.load(f)
    
    total = len(hotpot_data)
    print(f"Loaded {total} questions")
    print("-"*80)
    
    # Process in batches
    all_results = []
    start_time = time.time()
    
    for batch_start in range(0, total, CONCURRENCY):
        batch_end = min(batch_start + CONCURRENCY, total)
        batch_items = hotpot_data[batch_start:batch_end]
        
        print(f"\n>>> Batch {batch_start//CONCURRENCY + 1}: Questions {batch_start+1}-{batch_end}")
        
        batch_results = await run_batch(pipeline, batch_items, batch_start, total)
        all_results.extend(batch_results)
        
        # Progress summary
        elapsed = time.time() - start_time
        success_count = sum(1 for r in all_results if r['success'])
        error_count = len(all_results) - success_count
        
        print(f"    Completed: {len(all_results)}/{total} | Success: {success_count} | Errors: {error_count} | Time: {elapsed:.0f}s")
    
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
    print(f"Total Questions: {total}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Average Time per Question: {avg_time:.2f}s")
    print(f"Total Passages Used: {total_passages}")
    print(f"Average Passages per Question: {avg_passages:.1f}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Save results for separate evaluation
    output = {
        'config': {
            'concurrency': CONCURRENCY,
            'top_k': 3,
            'bm25_weight': 0.4,
            'dense_weight': 0.6
        },
        'summary': {
            'total_questions': total,
            'successful': success_count,
            'errors': error_count,
            'total_time': total_time,
            'avg_time_per_question': avg_time,
            'total_passages': total_passages,
            'avg_passages_per_question': avg_passages,
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    # Convert numpy types to native Python types
    output = convert_to_serializable(output)
    
    output_file = 'test_new_pipeline_200_results_v2.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
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
    asyncio.run(run_test())
