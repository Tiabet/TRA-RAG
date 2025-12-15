#!/usr/bin/env python3
"""
Test New Multi-hop Pipeline v5 (Path Reranking) on 200 Questions
================================================================
Unified pipeline test script for HotpotQA and MuSiQue datasets.

Usage:
    python test_pipeline_v5_200.py                     # Default: MuSiQue
    python test_pipeline_v5_200.py --dataset hotpotqa  # HotpotQA
    python test_pipeline_v5_200.py --dataset musique   # MuSiQue
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

from new_multihop_pipeline_v5 import NewMultihopPipelineV5, MetadataLinkerV5
from hybrid_path_retriever import HybridPathRetriever


# Concurrency settings
CONCURRENCY = 100 

# Dataset configurations
DATASET_CONFIGS = {
    'hotpotqa': {
        'data_path': 'HotpotQA/hotpotqa_sample_200.json',
        'metadata_json': 'HotpotQA/hotpotqa_sample_200_metadata.json',
        'bm25_index': 'HotpotQA/bm25_index',
        'path_embeddings': 'HotpotQA/path_embeddings.npz',
        'result_path': 'Results/test_hotpot_v5_200_results.json',
    },
    'musique': {
        'data_path': 'MuSiQue/musique_sample_200.json',
        'metadata_json': 'MuSiQue/musique_sample_200_metadata.json',
        'bm25_index': 'MuSiQue/bm25_index',
        'path_embeddings': 'MuSiQue/path_embeddings.npz',
        'result_path': 'Results/test_musique_v5_200_results_v1.json',
    }
}


async def process_single_question(pipeline, item, idx, total):
    """Process a single question and return result."""
    question = item['question']
    gold_answer = item['answer']
    qid = item.get('_id', f'q{idx}')
    
    start = time.time()
    
    try:
        # V5 pipeline uses .run()
        result = await pipeline.run(question)
        elapsed = time.time() - start
        
        predicted = result['predicted_answer']
        
        # Count passages used in final answer (from decomposition)
        num_passages = 0
        if 'decomposition' in result and 'subquestions' in result['decomposition']:
            seen_titles = set()
            for sq in result['decomposition']['subquestions']:
                if 'retrieved_passages' in sq and sq['retrieved_passages']:
                    for p in sq['retrieved_passages']:
                        if p['title'] not in seen_titles:
                            seen_titles.add(p['title'])
                            num_passages += 1

        print(f"[{idx+1:3d}/{total}] ✓ ({elapsed:.1f}s, {num_passages}p) {question[:50]}...")
        print(f"           Gold: {gold_answer}")
        print(f"           Pred: {predicted}")
        if predicted == "Decomposition failed.":
             print(f"           Error: {result.get('error')}")
        
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


async def main():
    parser = argparse.ArgumentParser(description="Test Pipeline V5")
    parser.add_argument("--dataset", type=str, default="musique", choices=["hotpotqa", "musique"], help="Dataset to test on")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--offset", type=int, default=0, help="Start index")
    parser.add_argument("--dense_weight", type=float, default=0.6, help="Weight for dense retrieval (0.0-1.0)")
    args = parser.parse_args()
    
    load_dotenv()
    
    # Check API Key
    if not os.getenv('ALICE_OPENAI_KEY'):
        print("Error: ALICE_OPENAI_KEY not found in .env")
        return

    config = DATASET_CONFIGS[args.dataset]
    print(f"Configuration: {args.dataset.upper()}")
    print(f"- Data: {config['data_path']}")
    print(f"- Metadata: {config['metadata_json']}")
    print(f"- Result: {config['result_path']}")
    print(f"- Dense Weight: {args.dense_weight} (BM25 Weight: {1.0 - args.dense_weight:.2f})")
    
    # Initialize Clients
    chat_client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    # Initialize Components
    print("\nInitializing Components...")
    
    # 1. Retriever (Hybrid Path Retriever)
    retriever = HybridPathRetriever(
        bm25_index_path=config['bm25_index'],
        embeddings_path=config['path_embeddings'],
        dense_weight=args.dense_weight,
        bm25_weight=1.0 - args.dense_weight
    )
    
    # 2. Linker (V5)
    linker = MetadataLinkerV5(config['metadata_json'])
    
    # 3. Pipeline (V5)
    pipeline = NewMultihopPipelineV5(
        client=chat_client,
        retriever=retriever,
        linker=linker,
        hotpotqa_path=config['data_path'],
        top_k=3,
        verbose=False
    )
    
    # Load Data
    print(f"Loading dataset from {config['data_path']}...")
    with open(config['data_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Apply limit/offset
    if args.limit:
        data = data[args.offset : args.offset + args.limit]
    else:
        data = data[args.offset:]
        
    print(f"Processing {len(data)} questions...")
    
    # Run Batch
    start_time = time.time()
    results = []
    
    # Semaphore for concurrency
    sem = asyncio.Semaphore(CONCURRENCY)
    
    async def bound_process(item, idx):
        async with sem:
            return await process_single_question(pipeline, item, idx, len(data))

    tasks = [bound_process(item, i) for i, item in enumerate(data)]
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    # Calculate Stats
    successful = [r for r in results if r['success']]
    avg_time = total_time / len(data) if data else 0
    total_passages = sum(r['num_passages'] for r in successful)
    avg_passages = total_passages / len(successful) if successful else 0
    
    print(f"\nDone in {total_time:.1f}s")
    print(f"Avg time per question: {avg_time:.1f}s")
    print(f"Avg passages used: {avg_passages:.1f}")
    
    # Save Results
    output = {
        'config': {
            'pipeline_version': 'v5',
            'dataset': args.dataset,
            'mode': 'path_reranking_zscore',
            'concurrency': CONCURRENCY,
            'top_k': 3,
            'dense_weight': args.dense_weight,
            'bm25_weight': 1.0 - args.dense_weight
        },
        'summary': {
            'total_questions': len(data),
            'successful': len(successful),
            'errors': len(data) - len(successful),
            'total_time': total_time,
            'avg_time_per_question': avg_time,
            'total_passages': total_passages,
            'avg_passages_per_question': avg_passages,
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }
    
    os.makedirs(os.path.dirname(config['result_path']), exist_ok=True)
    with open(config['result_path'], 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        
    print(f"Results saved to {config['result_path']}")

if __name__ == "__main__":
    asyncio.run(main())
