"""
Test Query Decomposition
==========================
Test the query decomposition module with sample HotpotQA questions.
"""

import asyncio
import json
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

from query_decomposition import decompose_query, get_execution_order

# Load environment
load_dotenv()


async def test_with_hotpot_samples(sample_file: str, num_samples: int = 10):
    """
    Test decomposition with HotpotQA sample questions.
    
    Args:
        sample_file: Path to hotpotqa_sample_200.json
        num_samples: Number of samples to test
    """
    # Initialize client
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    # Load samples
    with open(sample_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    # Select diverse samples (bridge and comparison)
    test_samples = []
    
    # Get some bridge questions
    bridge_samples = [s for s in samples if s.get('type') == 'bridge'][:num_samples//2]
    test_samples.extend(bridge_samples)
    
    # Get some comparison questions
    comparison_samples = [s for s in samples if s.get('type') == 'comparison'][:num_samples//2]
    test_samples.extend(comparison_samples)
    
    # If not enough, just take first num_samples
    if len(test_samples) < num_samples:
        test_samples = samples[:num_samples]
    
    results = []
    
    print(f"\n{'='*100}")
    print(f"Testing Query Decomposition with {len(test_samples)} HotpotQA samples")
    print(f"{'='*100}\n")
    
    for idx, sample in enumerate(test_samples, 1):
        query = sample['question']
        q_type = sample.get('type', 'unknown')
        q_id = sample.get('_id', f'Q{idx}')
        
        print(f"\n{'='*100}")
        print(f"Test {idx}/{len(test_samples)} [ID: {q_id}, Type: {q_type}]")
        print(f"{'='*100}")
        print(f"Question: {query}")
        print(f"Gold Answer: {sample.get('answer', 'N/A')}")
        
        # Decompose
        result = await decompose_query(client, query)
        
        if result['success']:
            decomp = result['decomposition']
            
            print(f"\n✅ Decomposition Successful")
            print(f"Detected Type: {decomp.question_type} (Gold: {q_type})")
            print(f"Reasoning: {decomp.reasoning}")
            print(f"\nSub-Questions ({len(decomp.subquestions)}):")
            
            for sq in decomp.subquestions:
                deps = f" [depends on: {', '.join(sq.depends_on)}]" if sq.depends_on else " [independent]"
                print(f"\n  {sq.id}: {sq.question}{deps}")
                print(f"  └─ Reasoning: {sq.reasoning}")
            
            # Show execution order
            try:
                execution_order = get_execution_order(decomp)
                print(f"\nExecution Order:")
                for batch_idx, batch in enumerate(execution_order, 1):
                    parallel = " (can execute in parallel)" if len(batch) > 1 else ""
                    print(f"  Batch {batch_idx}: {', '.join(batch)}{parallel}")
            except Exception as e:
                print(f"\n⚠️  Error getting execution order: {e}")
            
            # Check type match
            type_match = decomp.question_type == q_type
            print(f"\nType Match: {'✅' if type_match else '❌'} (Detected: {decomp.question_type}, Gold: {q_type})")
            
            results.append({
                'question_id': q_id,
                'question': query,
                'gold_type': q_type,
                'gold_answer': sample.get('answer', 'N/A'),
                'success': True,
                'detected_type': decomp.question_type,
                'type_match': type_match,
                'num_subquestions': len(decomp.subquestions),
                'subquestions': [sq.to_dict() for sq in decomp.subquestions],
                'execution_order': execution_order if 'execution_order' in locals() else []
            })
            
        else:
            print(f"\n❌ Decomposition Failed")
            print(f"Error: {result['error']}")
            if result.get('raw_response'):
                print(f"Raw response: {result['raw_response'][:500]}...")
            
            results.append({
                'question_id': q_id,
                'question': query,
                'gold_type': q_type,
                'success': False,
                'error': result['error']
            })
    
    # Summary statistics
    print(f"\n\n{'='*100}")
    print("SUMMARY STATISTICS")
    print(f"{'='*100}\n")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    
    if successful:
        type_matches = [r for r in successful if r.get('type_match', False)]
        print(f"\nType Detection Accuracy: {len(type_matches)}/{len(successful)} ({len(type_matches)/len(successful)*100:.1f}%)")
        
        # Average sub-questions
        avg_sqs = sum(r['num_subquestions'] for r in successful) / len(successful)
        print(f"Average Sub-Questions: {avg_sqs:.1f}")
        
        # By type
        bridge_results = [r for r in successful if r['detected_type'] == 'bridge']
        comparison_results = [r for r in successful if r['detected_type'] == 'comparison']
        
        print(f"\nBy Detected Type:")
        print(f"  Bridge: {len(bridge_results)} questions, avg {sum(r['num_subquestions'] for r in bridge_results)/len(bridge_results):.1f} SQs" if bridge_results else "  Bridge: 0 questions")
        print(f"  Comparison: {len(comparison_results)} questions, avg {sum(r['num_subquestions'] for r in comparison_results)/len(comparison_results):.1f} SQs" if comparison_results else "  Comparison: 0 questions")
    
    # Save results
    output_file = 'test_decomposition_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'type_matches': len([r for r in successful if r.get('type_match', False)]) if successful else 0
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {output_file}")
    
    return results


async def main():
    """Main test function"""
    sample_file = "HotpotQA/hotpotqa_sample_200.json"
    
    if not os.path.exists(sample_file):
        print(f"❌ Sample file not found: {sample_file}")
        print("Please ensure the file exists.")
        return
    
    # Test with 10 samples
    await test_with_hotpot_samples(sample_file, num_samples=10)


if __name__ == "__main__":
    asyncio.run(main())
