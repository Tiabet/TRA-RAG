"""
Detailed Query Decomposition Test
===================================
Tests query decomposition with FULL sub-question details visible.
Tests 5 questions from each dataset (15 total).
"""

import asyncio
import json
import sys
import io
from openai import AsyncOpenAI
from query_decomposition import decompose_query
from dotenv import load_dotenv
import os

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_samples():
    """Load samples from each dataset"""
    
    # HotpotQA: 3 bridge + 2 comparison
    with open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8') as f:
        hotpotqa = json.load(f)
    hotpot_bridge = [q for q in hotpotqa if q.get('type') == 'bridge'][:3]
    hotpot_comp = [q for q in hotpotqa if q.get('type') == 'comparison'][:2]
    
    # 2WikiMultihopQA: 2 compositional, 1 comparison, 1 bridge_comparison, 1 inference
    with open('2WikiMultihopQA/2wikimultihopqa_sample_200.json', 'r', encoding='utf-8') as f:
        wiki2 = json.load(f)
    wiki2_samples = []
    for qtype, count in [('compositional', 2), ('comparison', 1), ('bridge_comparison', 1), ('inference', 1)]:
        wiki2_samples.extend([q for q in wiki2 if q.get('type') == qtype][:count])
    
    # MuSiQue: 2-hop, 3-hop, 4-hop
    with open('MuSiQue/musique_qa_sample_200.json', 'r', encoding='utf-8') as f:
        musique = json.load(f)
    musique_2hop = [q for q in musique if len(q.get('question_decomposition', [])) == 2][:2]
    musique_3hop = [q for q in musique if len(q.get('question_decomposition', [])) == 3][:2]
    musique_4hop = [q for q in musique if len(q.get('question_decomposition', [])) == 4][:1]
    
    return {
        'HotpotQA': [(q['question'], q.get('type', 'unknown')) for q in hotpot_bridge + hotpot_comp],
        '2WikiMultihopQA': [(q['question'], q.get('type', 'unknown')) for q in wiki2_samples],
        'MuSiQue': [(q['question'], f"{len(q.get('question_decomposition', []))}-hop") for q in musique_2hop + musique_3hop + musique_4hop]
    }


async def test_one_question(client, question, label, original_type, show_reasoning=True):
    """Test and display detailed decomposition"""
    print(f"\n{'='*100}")
    print(f"[{label}] Original Type: {original_type}")
    print(f"{'='*100}")
    print(f"Question: {question}")
    print()
    
    result = await decompose_query(client, question)
    
    if result['success']:
        decomp = result['decomposition']
        
        print(f"✓ SUCCESS")
        print(f"  Predicted Type: {decomp.question_type}")
        print(f"  Total Sub-Questions: {len(decomp.subquestions)}")
        
        if show_reasoning:
            print(f"  Reasoning: {decomp.reasoning}")
        
        print(f"\nSub-Question Breakdown:")
        for sq in decomp.subquestions:
            if sq.depends_on:
                deps_str = f" (depends on: {', '.join(sq.depends_on)})"
            else:
                deps_str = " (INDEPENDENT - can run in parallel)"
            
            print(f"\n  [{sq.id}]{deps_str}")
            print(f"    Q: {sq.question}")
            print(f"    Why: {sq.reasoning}")
        
        # Check placeholder usage
        has_placeholders = any("[SQ" in sq.question for sq in decomp.subquestions)
        if has_placeholders:
            print(f"\n  ✓ Uses [SQ{{N}}_Answer] placeholders correctly")
        
        return True
    else:
        print(f"✗ FAILED: {result.get('error', 'Unknown error')}")
        return False


async def main():
    """Run detailed test"""
    
    # Load environment
    load_dotenv()
    
    # Initialize OpenAI client
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    print("="*100)
    print("DETAILED QUERY DECOMPOSITION TEST")
    print("="*100)
    print("Testing 15 real questions with full sub-question details")
    print()
    
    # Load samples
    samples = load_samples()
    
    print("Dataset Distribution:")
    for dataset, questions in samples.items():
        print(f"  {dataset}: {len(questions)} questions")
    print()
    
    # Test each dataset
    results = {}
    
    for dataset_name in ['HotpotQA', '2WikiMultihopQA', 'MuSiQue']:
        print(f"\n{'#'*100}")
        print(f"# {dataset_name}")
        print(f"{'#'*100}")
        
        dataset_results = []
        for idx, (question, qtype) in enumerate(samples[dataset_name], 1):
            label = f"{dataset_name}_{idx:02d}"
            success = await test_one_question(client, question, label, qtype)
            dataset_results.append(success)
            await asyncio.sleep(0.3)  # Rate limiting
        
        results[dataset_name] = dataset_results
    
    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    
    total_success = sum(sum(results[ds]) for ds in results)
    total_tests = sum(len(results[ds]) for ds in results)
    
    print(f"\nOverall: {total_success}/{total_tests} ({total_success/total_tests*100:.1f}%)")
    
    for dataset_name, dataset_results in results.items():
        success_count = sum(dataset_results)
        total_count = len(dataset_results)
        print(f"  {dataset_name}: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if total_success == total_tests:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ {total_tests - total_success} test(s) failed")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
