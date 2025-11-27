#!/usr/bin/env python3
"""
Debug Test: Compare V2 (Metadata) vs V3 (Original) on specific questions
=========================================================================
Runs both pipelines on a few questions with full logging enabled.
"""

import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

from new_multihop_pipeline import NewMultihopPipeline
from new_multihop_pipeline_v3 import NewMultihopPipelineV3
from hybrid_path_retriever import HybridPathRetriever
from llm_logger import init_logger, finalize_log


async def run_comparison(question_indices: list = None):
    """
    Run both V2 and V3 pipelines on specified questions for comparison.
    
    Args:
        question_indices: List of question indices to test (0-based). 
                         If None, uses default problematic cases.
    """
    load_dotenv()
    
    # Default: questions where V2 was correct but V3 was wrong
    if question_indices is None:
        question_indices = [0, 2, 5, 10, 21]  # First few for quick test
    
    print("="*80)
    print("Debug Comparison: V2 (Metadata) vs V3 (Original Passages)")
    print("="*80)
    print(f"Testing questions: {question_indices}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize logger
    logger = init_logger()
    print(f"Logging to: {logger.log_file}")
    
    # Initialize OpenAI client
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    # Initialize retriever (shared)
    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6
    )
    
    # Initialize both pipelines
    pipeline_v2 = NewMultihopPipeline(
        client=client,
        retriever=retriever,
        top_k=3,
        verbose=True
    )
    
    pipeline_v3 = NewMultihopPipelineV3(
        client=client,
        retriever=retriever,
        top_k=3,
        verbose=True
    )
    
    # Load HotpotQA sample
    with open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8') as f:
        hotpot_data = json.load(f)
    
    results = []
    
    for idx in question_indices:
        if idx >= len(hotpot_data):
            print(f"Skipping index {idx}: out of range")
            continue
            
        item = hotpot_data[idx]
        question = item['question']
        gold_answer = item['answer']
        
        print(f"\n{'#'*80}")
        print(f"Question #{idx}: {question}")
        print(f"Gold Answer: {gold_answer}")
        print(f"{'#'*80}")
        
        # Run V2 (Metadata)
        print(f"\n{'='*40} V2 (Metadata) {'='*40}")
        result_v2 = await pipeline_v2.process_question(question)
        pred_v2 = result_v2.get('final_answer', 'ERROR')
        
        # Run V3 (Original)
        print(f"\n{'='*40} V3 (Original) {'='*40}")
        result_v3 = await pipeline_v3.process_question(question)
        pred_v3 = result_v3.get('final_answer', 'ERROR')
        
        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY for Question #{idx}")
        print(f"{'='*80}")
        print(f"Question: {question}")
        print(f"Gold:     {gold_answer}")
        print(f"V2 Pred:  {pred_v2}")
        print(f"V3 Pred:  {pred_v3}")
        
        results.append({
            'index': idx,
            'question': question,
            'gold_answer': gold_answer,
            'v2_prediction': pred_v2,
            'v3_prediction': pred_v3,
            'v2_success': result_v2.get('success', False),
            'v3_success': result_v3.get('success', False)
        })
    
    # Final summary
    print(f"\n{'#'*80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'#'*80}")
    
    for r in results:
        gold = r['gold_answer'].lower()
        v2_match = gold in r['v2_prediction'].lower() or r['v2_prediction'].lower() in gold
        v3_match = gold in r['v3_prediction'].lower() or r['v3_prediction'].lower() in gold
        
        status = ""
        if v2_match and v3_match:
            status = "Both ✓"
        elif v2_match and not v3_match:
            status = "V2 ✓ V3 ✗"
        elif not v2_match and v3_match:
            status = "V2 ✗ V3 ✓"
        else:
            status = "Both ✗"
        
        print(f"Q{r['index']:3d}: {status:12s} | Gold: {r['gold_answer'][:30]:30s} | V2: {r['v2_prediction'][:25]:25s} | V3: {r['v3_prediction'][:25]}")
    
    # Finalize logger
    log_file = finalize_log()
    print(f"\n✅ Full logs saved to: {log_file}")
    
    # Clean up
    pipeline_v2.close()
    pipeline_v3.close()
    
    return results


if __name__ == "__main__":
    import sys
    
    # Allow custom indices via command line
    if len(sys.argv) > 1:
        indices = [int(x) for x in sys.argv[1:]]
    else:
        # Default: test first 3 questions for quick debug
        indices = [0, 1, 2]
    
    asyncio.run(run_comparison(indices))
