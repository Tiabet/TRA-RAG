#!/usr/bin/env python3
"""
Test New Multi-hop Pipeline v4 (Metadata Expansion + Reranking) on 200 Questions
=================================================================================
Unified pipeline test script for HotpotQA and MuSiQue datasets.

Usage:
    python test_pipeline_v4_200.py                     # Default: HotpotQA
    python test_pipeline_v4_200.py --dataset hotpotqa  # HotpotQA
    python test_pipeline_v4_200.py --dataset musique   # MuSiQue
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

from new_multihop_pipeline_v4 import NewMultihopPipelineV4, MetadataLinker, PassageReranker
from hybrid_path_retriever import HybridPathRetriever


# Concurrency settings
CONCURRENCY = 100  # Reduced concurrency due to heavier processing (reranking)

# Dataset configurations
DATASET_CONFIGS = {
    'hotpotqa': {
        'data_path': 'HotpotQA/hotpotqa_sample_200.json',
        'metadata_json': 'HotpotQA/hotpotqa_sample_200_metadata.json',
        'passage_embeddings': 'HotpotQA/passage_embeddings.npz',
        'bm25_index': 'HotpotQA/bm25_index',
        'path_embeddings': 'HotpotQA/path_embeddings.npz',
        'result_path': 'Results/test_hotpot_v4_200_results.json',
    },
    'musique': {
        'data_path': 'MuSiQue/musique_sample_200.json',
        'metadata_json': 'MuSiQue/musique_sample_200_metadata.json',
        'passage_embeddings': 'MuSiQue/passage_embeddings.npz',
        'bm25_index': 'MuSiQue/bm25_index',
        'path_embeddings': 'MuSiQue/path_embeddings.npz',
        'result_path': 'Results/test_musique_v4_200_results_v3.json',
    }
}


async def process_single_question(pipeline, item, idx, total):
    """Process a single question and return result."""
    question = item['question']
    gold_answer = item['answer']
    qid = item.get('_id', f'q{idx}')
    
    start = time.time()
    
    try:
        # V4 pipeline uses .run() instead of .process_question()
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
    parser = argparse.ArgumentParser(description="Test Pipeline V4")
    parser.add_argument("--dataset", type=str, default="musique", choices=["hotpotqa", "musique"], help="Dataset to test on")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--offset", type=int, default=0, help="Start index")
    parser.add_argument("--dense_weight", type=float, default=1.0, help="Weight for dense retrieval (0.0-1.0)")
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
    print(f"- Passage Embeddings: {config['passage_embeddings']}")
    print(f"- Result: {config['result_path']}")
    print(f"- Dense Weight: {args.dense_weight} (BM25 Weight: {1.0 - args.dense_weight:.2f})")
    
    # Initialize Clients
    # We need two clients: one for Chat (Decomposition/Answering) and one for Embeddings (Reranking)
    
    chat_client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')  # Use Chat URL for pipeline
    )
    
    embed_client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_EMBED_URL') # Use Embed URL for reranker
    )
    
    # Initialize Components
    print("\nInitializing Components...")
    
    # 1. Retriever
    retriever = HybridPathRetriever(
        bm25_index_path=config['bm25_index'],
        embeddings_path=config['path_embeddings'],
        dense_weight=args.dense_weight,
        bm25_weight=1.0 - args.dense_weight
    )
    
    # 2. Linker
    linker = MetadataLinker(config['metadata_json'])
    
    # 3. Reranker
    if not os.path.exists(config['passage_embeddings']):
        print(f"Error: Passage embeddings file not found at {config['passage_embeddings']}")
        print("Please run generate_passage_embeddings.py first.")
        return
        
    reranker = PassageReranker(config['passage_embeddings'], embed_client)
    
    # 4. Pipeline
    pipeline = NewMultihopPipelineV4(
        client=chat_client,
        retriever=retriever,
        linker=linker,
        reranker=reranker,
        hotpotqa_path=config['data_path'],
        top_k=3,
        verbose=False  # Reduce verbosity for batch run
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
        
    total_questions = len(data)
    print(f"Processing {total_questions} questions...")
    
    # Run Batch
    results = []
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    async def sem_task(item, idx):
        async with semaphore:
            return await process_single_question(pipeline, item, idx, total_questions)
    
    tasks = [sem_task(item, i) for i, item in enumerate(data)]
    results = await asyncio.gather(*tasks)
    
    # Calculate Stats
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    avg_time = sum(r['time'] for r in results) / len(results) if results else 0
    
    print("\n" + "="*50)
    print("Results Summary")
    print("="*50)
    print(f"Total: {len(results)}")
    print(f"Success: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Avg Time: {avg_time:.2f}s")
    
    # Save Results
    os.makedirs(os.path.dirname(config['result_path']), exist_ok=True)
    with open(config['result_path'], 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nResults saved to {config['result_path']}")

if __name__ == "__main__":
    asyncio.run(main())
