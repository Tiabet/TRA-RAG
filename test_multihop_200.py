"""
Test Multi-hop Pipeline on 200 Questions
==========================================
Complete evaluation with detailed result tracking.
"""

import asyncio
import json
import time
from typing import Dict, List
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm

from metadata_db import MetadataDB
from multihop_pipeline import process_single_question

load_dotenv()


def calculate_exact_match(predicted: str, gold: str) -> bool:
    """
    Calculate exact match with normalization.
    """
    def normalize(text):
        if not text:
            return ""
        text = str(text).lower().strip()
        # Remove articles
        for article in ['the', 'a', 'an']:
            text = text.replace(f' {article} ', ' ')
        # Remove punctuation
        import string
        text = ''.join(c if c not in string.punctuation else ' ' for c in text)
        return ' '.join(text.split())
    
    return normalize(predicted) == normalize(gold)


def calculate_token_f1(predicted: str, gold: str) -> float:
    """
    Calculate token-level F1 score.
    """
    def get_tokens(text):
        if not text:
            return set()
        text = str(text).lower().strip()
        import string
        text = ''.join(c if c not in string.punctuation else ' ' for c in text)
        return set(text.split())
    
    pred_tokens = get_tokens(predicted)
    gold_tokens = get_tokens(gold)
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


async def process_question_with_safety(
    client: AsyncOpenAI,
    db: MetadataDB,
    question_data: Dict,
    question_idx: int,
    use_fts: bool = True,
    apply_llm_filter_stage1a: bool = True
) -> Dict:
    """
    Process a single question with thread-safety measures.
    Each question gets its own isolated processing to prevent data mixing.
    
    Args:
        client: AsyncOpenAI client (shared, but stateless)
        db: MetadataDB instance (shared, read-only)
        question_data: Question dict (isolated copy)
        question_idx: Index for tracking
        use_fts: Use FTS for retrieval
        apply_llm_filter_stage1a: Apply LLM filtering
        
    Returns:
        Complete result dict with all details
    """
    try:
        # Make a deep copy to prevent cross-contamination
        question_copy = {
            '_id': question_data.get('_id', f'Q{question_idx}'),
            'question': question_data['question'],
            'answer': question_data.get('answer', 'N/A'),
            'type': question_data.get('type', 'unknown'),
            'level': question_data.get('level', 'unknown'),
            'supporting_facts': question_data.get('supporting_facts', [])
        }
        
        # Process through pipeline
        result = await process_single_question(
            client, db, question_copy,
            use_fts, apply_llm_filter_stage1a,
            verbose=False  # Disable verbose for batch processing
        )
        
        # Extract detailed information
        if result['success']:
            decomposition = result.get('decomposition', {})
            
            # Collect all retrieved passage titles
            all_passage_titles = []
            for sq in decomposition.get('subquestions', []):
                passages = sq.get('retrieved_passages', [])
                for passage in passages:
                    title = passage.get('title', '')
                    if title and title not in all_passage_titles:
                        all_passage_titles.append(title)
            
            # Collect all extracted entities
            all_entities = []
            for sq in decomposition.get('subquestions', []):
                retrieval_info = sq.get('retrieval_info', {})
                entities = retrieval_info.get('extracted_entities', [])
                for entity in entities:
                    entity_name = entity.get('entity_name', '')
                    if entity_name:
                        all_entities.append({
                            'name': entity_name,
                            'types': entity.get('possible_types', []),
                            'role': entity.get('role', ''),
                            'subquestion_id': sq.get('id', '')
                        })
            
            # Calculate metrics
            predicted_answer = result.get('final_answer', '')
            gold_answer = question_copy['answer']
            
            exact_match = calculate_exact_match(predicted_answer, gold_answer)
            token_f1 = calculate_token_f1(predicted_answer, gold_answer)
            
            # Build complete result
            complete_result = {
                'question_id': question_copy['_id'],
                'question': question_copy['question'],
                'question_type': question_copy['type'],
                'question_level': question_copy['level'],
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'exact_match': exact_match,
                'token_f1': token_f1,
                'success': True,
                
                # Decomposition details
                'decomposition': {
                    'detected_type': result.get('question_type', 'unknown'),
                    'num_subquestions': len(decomposition.get('subquestions', [])),
                    'subquestions': [
                        {
                            'id': sq.get('id', ''),
                            'question': sq.get('question', ''),
                            'answer': sq.get('answer', ''),
                            'depends_on': sq.get('depends_on', []),
                            'reasoning': sq.get('reasoning', '')
                        }
                        for sq in decomposition.get('subquestions', [])
                    ]
                },
                
                # Retrieved passages
                'retrieved_passages': {
                    'titles': all_passage_titles,
                    'count': len(all_passage_titles),
                    'by_subquestion': [
                        {
                            'subquestion_id': sq.get('id', ''),
                            'titles': [p.get('title', '') for p in sq.get('retrieved_passages', [])]
                        }
                        for sq in decomposition.get('subquestions', [])
                    ]
                },
                
                # Extracted entities
                'extracted_entities': {
                    'all': all_entities,
                    'count': len(all_entities),
                    'unique_names': list(set(e['name'] for e in all_entities))
                },
                
                # Supporting facts (gold)
                'gold_supporting_facts': question_copy.get('supporting_facts', []),
                
                # Timing
                'timing': result.get('timing', {}),
                
                # Processing metadata
                'processing_index': question_idx
            }
            
            return complete_result
        
        else:
            # Failed processing
            return {
                'question_id': question_copy['_id'],
                'question': question_copy['question'],
                'question_type': question_copy['type'],
                'gold_answer': question_copy['answer'],
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'processing_index': question_idx
            }
            
    except Exception as e:
        return {
            'question_id': question_data.get('_id', f'Q{question_idx}'),
            'question': question_data.get('question', ''),
            'success': False,
            'error': f"Exception: {str(e)}",
            'processing_index': question_idx
        }


async def evaluate_multihop_200(
    questions: List[Dict],
    max_workers: int = 30,
    use_fts: bool = True,
    apply_llm_filter_stage1a: bool = True
):
    """
    Evaluate multi-hop pipeline on 200 questions with high parallelism.
    
    Args:
        questions: List of question dicts
        max_workers: Number of parallel workers (30 recommended for no API limit)
        use_fts: Use FTS for retrieval
        apply_llm_filter_stage1a: Apply LLM filtering to Stage 1-A
    """
    print(f"\n{'='*100}")
    print(f"Multi-hop Pipeline Evaluation: 200 Questions")
    print(f"{'='*100}")
    print(f"Max Workers: {max_workers}")
    print(f"Use FTS: {use_fts}")
    print(f"LLM Filtering (Stage 1-A): {apply_llm_filter_stage1a}")
    print(f"{'='*100}\n")
    
    start_time = time.time()
    
    # Initialize clients (one per worker would be ideal, but shared is fine)
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    db = MetadataDB('metadata_v2.db')
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_with_semaphore(question_data, idx):
        async with semaphore:
            return await process_question_with_safety(
                client, db, question_data, idx,
                use_fts, apply_llm_filter_stage1a
            )
    
    # Process all questions with progress bar and checkpoint saving
    print(f"Processing {len(questions)} questions with {max_workers} workers...\n")
    print("Saving checkpoints every 10 questions to 'multihop_pipeline_200_checkpoint.json'\n")
    
    # Create tasks
    tasks = [
        process_with_semaphore(q, i)
        for i, q in enumerate(questions)
    ]
    
    # Execute with progress tracking and checkpoint saving
    results = []
    checkpoint_interval = 10
    checkpoint_file = 'multihop_pipeline_200_checkpoint.json'
    
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Questions"):
        result = await coro
        results.append(result)
        
        # Save checkpoint every 10 results
        if len(results) % checkpoint_interval == 0:
            # Sort by processing_index before saving
            sorted_results = sorted(results, key=lambda x: x.get('processing_index', 0))
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'completed': len(results),
                    'total': len(questions),
                    'results': sorted_results
                }, f, indent=2, ensure_ascii=False)
    
    # Sort results by processing_index to maintain order
    results.sort(key=lambda x: x.get('processing_index', 0))
    
    total_time = time.time() - start_time
    
    # Close connections
    db.close()
    await client.close()
    
    # Calculate statistics
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    if successful:
        exact_matches = [r for r in successful if r.get('exact_match', False)]
        avg_token_f1 = sum(r.get('token_f1', 0) for r in successful) / len(successful)
        
        # By question type
        bridge_results = [r for r in successful if r.get('question_type') == 'bridge']
        comparison_results = [r for r in successful if r.get('question_type') == 'comparison']
        
        # Timing statistics
        avg_total_time = sum(r['timing']['total'] for r in successful) / len(successful)
        avg_decomp_time = sum(r['timing']['decomposition'] for r in successful) / len(successful)
        avg_answering_time = sum(r['timing']['answering'] for r in successful) / len(successful)
        avg_synthesis_time = sum(r['timing']['synthesis'] for r in successful) / len(successful)
    
    # Print summary
    print(f"\n{'='*100}")
    print("EVALUATION SUMMARY")
    print(f"{'='*100}")
    print(f"\nTotal Questions: {len(results)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    
    if successful:
        print(f"\n{'='*100}")
        print("ANSWER QUALITY METRICS")
        print(f"{'='*100}")
        print(f"Exact Match (EM): {len(exact_matches)}/{len(successful)} ({len(exact_matches)/len(successful)*100:.1f}%)")
        print(f"Average Token F1: {avg_token_f1:.4f}")
        
        print(f"\n{'='*100}")
        print("BY QUESTION TYPE")
        print(f"{'='*100}")
        
        if bridge_results:
            bridge_em = [r for r in bridge_results if r.get('exact_match', False)]
            bridge_f1 = sum(r.get('token_f1', 0) for r in bridge_results) / len(bridge_results)
            print(f"\nBridge Questions ({len(bridge_results)}):")
            print(f"  Exact Match: {len(bridge_em)}/{len(bridge_results)} ({len(bridge_em)/len(bridge_results)*100:.1f}%)")
            print(f"  Token F1: {bridge_f1:.4f}")
        
        if comparison_results:
            comp_em = [r for r in comparison_results if r.get('exact_match', False)]
            comp_f1 = sum(r.get('token_f1', 0) for r in comparison_results) / len(comparison_results)
            print(f"\nComparison Questions ({len(comparison_results)}):")
            print(f"  Exact Match: {len(comp_em)}/{len(comparison_results)} ({len(comp_em)/len(comparison_results)*100:.1f}%)")
            print(f"  Token F1: {comp_f1:.4f}")
        
        print(f"\n{'='*100}")
        print("TIMING STATISTICS")
        print(f"{'='*100}")
        print(f"Total Wall Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Average Time per Question: {avg_total_time:.2f}s")
        print(f"Theoretical Sequential Time: {avg_total_time * len(results):.2f}s ({avg_total_time * len(results)/60:.2f} minutes)")
        print(f"Speedup Factor: {(avg_total_time * len(results)) / total_time:.2f}x")
        
        print(f"\nAverage Timing Breakdown:")
        print(f"  Decomposition: {avg_decomp_time:.2f}s ({avg_decomp_time/avg_total_time*100:.1f}%)")
        print(f"  Answering: {avg_answering_time:.2f}s ({avg_answering_time/avg_total_time*100:.1f}%)")
        print(f"  Synthesis: {avg_synthesis_time:.2f}s ({avg_synthesis_time/avg_total_time*100:.1f}%)")
    
    if failed:
        print(f"\n{'='*100}")
        print(f"FAILED QUESTIONS ({len(failed)})")
        print(f"{'='*100}")
        for i, f in enumerate(failed[:5], 1):  # Show first 5
            print(f"\n{i}. {f.get('question_id', 'Unknown')}")
            print(f"   Question: {f.get('question', '')[:80]}...")
            print(f"   Error: {f.get('error', 'Unknown')}")
    
    # Save detailed results
    output_file = 'multihop_pipeline_200_results.json'
    print(f"\n{'='*100}")
    print(f"Saving detailed results to {output_file}...")
    print(f"{'='*100}")
    
    output_data = {
        'metadata': {
            'total_questions': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'max_workers': max_workers,
            'total_time': total_time,
            'use_fts': use_fts,
            'apply_llm_filter_stage1a': apply_llm_filter_stage1a
        },
        'summary': {
            'exact_match': len(exact_matches) if successful else 0,
            'exact_match_rate': len(exact_matches)/len(successful) if successful else 0,
            'avg_token_f1': avg_token_f1 if successful else 0,
            'avg_time_per_question': avg_total_time if successful else 0,
            'speedup_factor': (avg_total_time * len(results)) / total_time if successful else 0
        },
        'by_type': {
            'bridge': {
                'count': len(bridge_results) if successful else 0,
                'exact_match': len([r for r in bridge_results if r.get('exact_match', False)]) if successful and bridge_results else 0,
                'avg_token_f1': sum(r.get('token_f1', 0) for r in bridge_results) / len(bridge_results) if successful and bridge_results else 0
            },
            'comparison': {
                'count': len(comparison_results) if successful else 0,
                'exact_match': len([r for r in comparison_results if r.get('exact_match', False)]) if successful and comparison_results else 0,
                'avg_token_f1': sum(r.get('token_f1', 0) for r in comparison_results) / len(comparison_results) if successful and comparison_results else 0
            }
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved!")
    print(f"\nResult file includes:")
    print(f"  - Question and answer details")
    print(f"  - Query decomposition")
    print(f"  - Retrieved passage titles")
    print(f"  - Extracted entities")
    print(f"  - Timing information")
    print(f"  - All metadata")
    
    return output_data


async def main():
    """Main execution"""
    # Load questions
    questions_file = 'HotpotQA/hotpotqa_sample_200.json'
    
    if not os.path.exists(questions_file):
        print(f"❌ Questions file not found: {questions_file}")
        return
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions from {questions_file}")
    
    # Run evaluation with high parallelism
    # 30 workers: good balance for 200 questions (no API limit)
    results = await evaluate_multihop_200(
        questions,
        max_workers=30,  # High parallelism!
        use_fts=True,
        apply_llm_filter_stage1a=True
    )
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
