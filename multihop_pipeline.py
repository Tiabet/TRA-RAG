"""
Multi-hop Pipeline
===================
Complete end-to-end pipeline for multi-hop question answering.

Supports:
- Query Decomposition
- Batch-aware parallel sub-question answering (within a question)
- Multi-question parallel processing (across questions)
- Final answer synthesis
"""

import asyncio
import json
import time
from typing import Dict, List, Optional
from openai import AsyncOpenAI

from metadata_db import MetadataDB
from query_decomposition import decompose_query, QueryDecomposition
from sequential_answering import (
    answer_subquestions_sequential,
    synthesize_final_answer
)


async def process_single_question(
    client: AsyncOpenAI,
    db: MetadataDB,
    question_data: Dict,
    use_fts: bool = True,
    apply_llm_filter_stage1a: bool = True,
    verbose: bool = False
) -> Dict:
    """
    Process a single question through the complete pipeline.
    
    Pipeline:
    1. Query Decomposition
    2. Sequential answering (with batch-aware parallel execution)
    3. Final answer synthesis
    
    Args:
        client: AsyncOpenAI client
        db: MetadataDB instance
        question_data: Dict with 'question', '_id', 'answer', etc.
        use_fts: Use FTS for retrieval
        apply_llm_filter_stage1a: Apply LLM filtering to Stage 1-A
        verbose: Print progress
        
    Returns:
        Dict with results and timing information
    """
    question_id = question_data.get('_id', 'unknown')
    question = question_data['question']
    gold_answer = question_data.get('answer', 'N/A')
    
    start_time = time.time()
    
    try:
        if verbose:
            print(f"\n{'='*100}")
            print(f"Processing Question: {question_id}")
            print(f"{'='*100}")
            print(f"Question: {question}")
            print(f"Gold Answer: {gold_answer}")
        
        # Step 1: Query Decomposition
        if verbose:
            print(f"\nStep 1: Query Decomposition...")
        
        decomp_start = time.time()
        decomp_result = await decompose_query(client, question)
        decomp_time = time.time() - decomp_start
        
        if not decomp_result['success']:
            return {
                'question_id': question_id,
                'question': question,
                'gold_answer': gold_answer,
                'success': False,
                'error': f"Decomposition failed: {decomp_result.get('error', 'Unknown')}",
                'total_time': time.time() - start_time
            }
        
        decomposition = decomp_result['decomposition']
        
        if verbose:
            print(f"✅ Decomposed into {len(decomposition.subquestions)} sub-questions")
            print(f"   Type: {decomposition.question_type}")
            print(f"   Time: {decomp_time:.2f}s")
        
        # Step 2: Answer sub-questions
        if verbose:
            print(f"\nStep 2: Answering Sub-Questions...")
        
        answering_start = time.time()
        answering_result = await answer_subquestions_sequential(
            client, db, decomposition,
            use_fts, apply_llm_filter_stage1a,
            verbose=verbose
        )
        answering_time = time.time() - answering_start
        
        if not answering_result['success']:
            return {
                'question_id': question_id,
                'question': question,
                'gold_answer': gold_answer,
                'success': False,
                'error': f"Answering failed: {answering_result.get('error', 'Unknown')}",
                'decomposition': decomposition.to_dict(),
                'total_time': time.time() - start_time
            }
        
        if verbose:
            print(f"\n✅ Answered {len(decomposition.subquestions)} sub-questions")
            print(f"   Batches: {answering_result.get('num_batches', 0)}")
            print(f"   Time: {answering_time:.2f}s")
        
        # Step 3: Synthesize final answer
        if verbose:
            print(f"\nStep 3: Final Answer Synthesis...")
        
        synthesis_start = time.time()
        final_answer = await synthesize_final_answer(client, decomposition)
        synthesis_time = time.time() - synthesis_start
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*100}")
            print(f"FINAL ANSWER: {final_answer}")
            print(f"Gold Answer:  {gold_answer}")
            print(f"{'='*100}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"  - Decomposition: {decomp_time:.2f}s")
            print(f"  - Answering: {answering_time:.2f}s")
            print(f"  - Synthesis: {synthesis_time:.2f}s")
        
        return {
            'question_id': question_id,
            'question': question,
            'question_type': decomposition.question_type,
            'gold_answer': gold_answer,
            'final_answer': final_answer,
            'success': True,
            'decomposition': decomposition.to_dict(),
            'timing': {
                'total': total_time,
                'decomposition': decomp_time,
                'answering': answering_time,
                'synthesis': synthesis_time
            }
        }
        
    except Exception as e:
        return {
            'question_id': question_id,
            'question': question,
            'gold_answer': gold_answer,
            'success': False,
            'error': str(e),
            'total_time': time.time() - start_time
        }


async def process_multiple_questions(
    client: AsyncOpenAI,
    db: MetadataDB,
    questions: List[Dict],
    max_workers: int = 100,
    batch_size: int = 42,
    use_fts: bool = True,
    apply_llm_filter_stage1a: bool = True,
    verbose: bool = False
) -> List[Dict]:
    """
    Process multiple questions in parallel with batching.
    
    Args:
        client: AsyncOpenAI client
        db: MetadataDB instance
        questions: List of question dicts
        max_workers: Maximum concurrent questions (default: 100, no API limit!)
        batch_size: Number of questions per batch (default: 42)
        use_fts: Use FTS for retrieval
        apply_llm_filter_stage1a: Apply LLM filtering to Stage 1-A
        verbose: Print progress for each question
        
    Returns:
        List of result dicts
    """
    print(f"\n{'='*100}")
    print(f"Multi-hop Pipeline: Processing {len(questions)} questions")
    print(f"Batch Size: {batch_size} questions/batch")
    print(f"Max Workers: {max_workers} (parallel question processing)")
    print(f"Total Batches: {(len(questions) + batch_size - 1) // batch_size}")
    print(f"{'='*100}\n")
    
    start_time = time.time()
    all_results = []
    
    # Process in batches
    for batch_idx in range(0, len(questions), batch_size):
        batch = questions[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        total_batches = (len(questions) + batch_size - 1) // batch_size
        
        print(f"\n{'='*100}")
        print(f"Processing Batch {batch_num}/{total_batches} ({len(batch)} questions)")
        print(f"{'='*100}")
        
        # Create semaphore to limit concurrent questions within this batch
        semaphore = asyncio.Semaphore(max_workers)
        
        async def process_with_semaphore(question_data):
            async with semaphore:
                return await process_single_question(
                    client, db, question_data,
                    use_fts, apply_llm_filter_stage1a,
                    verbose
                )
        
        # Process all questions in this batch in parallel (with max_workers limit)
        batch_results = await asyncio.gather(
            *[process_with_semaphore(q) for q in batch],
            return_exceptions=True
        )
        
        # Handle exceptions
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                all_results.append({
                    'question_id': batch[i].get('_id', f'Q{batch_idx + i}'),
                    'question': batch[i]['question'],
                    'success': False,
                    'error': str(result)
                })
            else:
                all_results.append(result)
        
        # Batch summary
        batch_successful = sum(1 for r in batch_results if not isinstance(r, Exception) and r.get('success', False))
        print(f"\nBatch {batch_num} Complete: {batch_successful}/{len(batch)} successful")
    
    total_time = time.time() - start_time
    
    # Summary statistics
    successful = [r for r in all_results if r.get('success', False)]
    failed = [r for r in all_results if not r.get('success', False)]
    
    print(f"\n{'='*100}")
    print("PIPELINE SUMMARY")
    print(f"{'='*100}")
    print(f"Total Questions: {len(questions)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time/Question: {total_time/len(questions):.2f}s")
    
    if successful:
        avg_decomp = sum(r['timing']['decomposition'] for r in successful) / len(successful)
        avg_answering = sum(r['timing']['answering'] for r in successful) / len(successful)
        avg_synthesis = sum(r['timing']['synthesis'] for r in successful) / len(successful)
        
        print(f"\nAverage Timing Breakdown:")
        print(f"  Decomposition: {avg_decomp:.2f}s")
        print(f"  Answering: {avg_answering:.2f}s")
        print(f"  Synthesis: {avg_synthesis:.2f}s")
    
    return all_results


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def test_single_question():
        """Test pipeline with a single question"""
        
        client = AsyncOpenAI(
            api_key=os.getenv('ALICE_OPENAI_KEY'),
            base_url=os.getenv('ALICE_CHAT_URL')
        )
        
        db = MetadataDB('metadata_v2.db')
        
        # Test question
        question_data = {
            '_id': 'test_001',
            'question': "The Bee Cliff in northeast Tennessee overlooks a river that is how many miles long?",
            'answer': "78.5 miles",
            'type': 'bridge'
        }
        
        result = await process_single_question(
            client, db, question_data,
            use_fts=True,
            apply_llm_filter_stage1a=True,
            verbose=True
        )
        
        db.close()
        
        return result
    
    async def test_multiple_questions():
        """Test pipeline with multiple questions"""
        
        client = AsyncOpenAI(
            api_key=os.getenv('ALICE_OPENAI_KEY'),
            base_url=os.getenv('ALICE_CHAT_URL')
        )
        
        db = MetadataDB('metadata_v2.db')
        
        # Load sample questions
        with open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8') as f:
            all_questions = json.load(f)
        
        # Test with first 5 questions
        test_questions = all_questions[:5]
        
        results = await process_multiple_questions(
            client, db, test_questions,
            max_workers=100,  # Process 100 questions in parallel
            batch_size=42,    # 42 questions per batch
            use_fts=True,
            apply_llm_filter_stage1a=True,
            verbose=False  # Set to True to see detailed progress
        )
        
        db.close()
        
        return results
    
    # Run tests
    print("="*100)
    print("Testing Multi-hop Pipeline")
    print("="*100)
    
    # Test single question
    print("\n[TEST 1] Single Question Test\n")
    result = asyncio.run(test_single_question())
    
    # Test multiple questions
    print("\n\n[TEST 2] Multiple Questions Test (5 questions in parallel)\n")
    results = asyncio.run(test_multiple_questions())
