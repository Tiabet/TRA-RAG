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
from hybrid_path_retriever import HybridPathRetriever

# Load environment variables
load_dotenv()

# Concurrency settings
CONCURRENCY = 50  # Adjust based on rate limits

async def process_single_question(pipeline, item, idx, total):
    """Process a single question and return result."""
    question = item['question']
    gold_answer = item['answer']
    qid = item.get('_id', f'q{idx}')
    
    start = time.time()
    
    try:
        result = await pipeline.process_question(question)
        elapsed = time.time() - start
        
        retrieved_docs = set()
        final_answer = None
        
        if result['success']:
            final_answer = result['final_answer']
            # Extract all retrieved passages from decomposition
            if 'decomposition' in result and 'subquestions' in result['decomposition']:
                for sq in result['decomposition']['subquestions']:
                    for p in sq.get('retrieved_passages', []):
                        retrieved_docs.add(p['title'])
            
            print(f"[{idx+1:3d}/{total}] ✓ ({elapsed:.1f}s) {question[:50]}...")
            
            return {
                'question_id': qid,
                'question': question,
                'supporting_facts': [f[0] for f in item.get('supporting_facts', [])],
                'retrieved_docs': list(retrieved_docs),
                'final_answer': final_answer,
                'gold_answer': gold_answer,
                'success': True,
                'time': elapsed
            }
        else:
            print(f"[{idx+1:3d}/{total}] ✗ ({elapsed:.1f}s) Error: {result.get('error')[:50]}...")
            return {
                'question_id': qid,
                'question': question,
                'supporting_facts': [f[0] for f in item.get('supporting_facts', [])],
                'retrieved_docs': [],
                'final_answer': None,
                'gold_answer': gold_answer,
                'success': False,
                'error': result.get('error'),
                'time': elapsed
            }
            
    except Exception as e:
        elapsed = time.time() - start
        print(f"[{idx+1:3d}/{total}] ✗ ({elapsed:.1f}s) Exception: {str(e)[:50]}...")
        return {
            'question_id': qid,
            'question': question,
            'supporting_facts': [f[0] for f in item.get('supporting_facts', [])],
            'retrieved_docs': [],
            'final_answer': None,
            'gold_answer': gold_answer,
            'success': False,
            'error': str(e),
            'time': elapsed
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
                'question_id': item.get('_id', f'q{idx}'),
                'question': item['question'],
                'supporting_facts': [f[0] for f in item.get('supporting_facts', [])],
                'retrieved_docs': [],
                'final_answer': None,
                'gold_answer': item['answer'],
                'success': False,
                'error': str(r),
                'time': 0
            })
        else:
            processed_results.append(r)
    
    return processed_results

async def run_musique_retrieval():
    # Configuration
    DATA_PATH = 'MuSiQue/musique_sample_200.json'
    RESULT_PATH = 'Results/musique_retrieval_analysis_results.json'
    
    print("="*80)
    print(f"Running MuSiQue Pipeline on ALL 200 questions (Concurrent)")
    print("="*80)

    # Initialize components
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    retriever = HybridPathRetriever(
        bm25_index_path='MuSiQue/bm25_index',
        embeddings_path='MuSiQue/path_embeddings.npz',
        bm25_weight=0.4,
        dense_weight=0.6
    )
    
    pipeline = NewMultihopPipelineV3(
        client=client,
        retriever=retriever,
        hotpotqa_path=DATA_PATH,
        db_path='MuSiQue/metadata_v3.db',
        top_k=3,
        verbose=False 
    )
    
    # Load Data
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_questions = len(data)
    print(f"Loaded {total_questions} questions.")
    
    all_results = []
    start_time = time.time()
    
    # Process in batches
    for batch_start in range(0, total_questions, CONCURRENCY):
        batch_end = min(batch_start + CONCURRENCY, total_questions)
        batch_items = data[batch_start:batch_end]
        
        print(f"\n>>> Batch {batch_start//CONCURRENCY + 1}: Questions {batch_start+1}-{batch_end}")
        
        batch_results = await run_batch(pipeline, batch_items, batch_start, total_questions)
        all_results.extend(batch_results)
        
        # Save intermediate results
        os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
        with open(RESULT_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"    (Saved progress to {RESULT_PATH})")

    # Final Save
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    total_time = time.time() - start_time
    print(f"\nFinal results saved to {RESULT_PATH}")
    print(f"Total Time: {total_time:.1f}s")
    
    pipeline.close()

if __name__ == "__main__":
    asyncio.run(run_musique_retrieval())
