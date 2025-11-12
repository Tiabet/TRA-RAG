"""
Comprehensive Query Decomposition Test
========================================
Tests the updated type-agnostic decomposition on 30 real questions:
- HotpotQA: 10 questions (bridge + comparison mix)
- 2WikiMultihopQA: 10 questions (all 4 types)
- MuSiQue: 10 questions (2-4 hops mix)

Uses ACTUAL questions from each dataset.
"""

import asyncio
import json
from openai import AsyncOpenAI
from query_decomposition import decompose_query
from dotenv import load_dotenv
import os
from collections import defaultdict


def load_hotpotqa_samples(n_per_type=5):
    """Load HotpotQA samples with type distribution"""
    with open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Separate by type
    bridge = [q for q in data if q.get('type') == 'bridge']
    comparison = [q for q in data if q.get('type') == 'comparison']
    
    # Sample evenly
    samples = bridge[:n_per_type] + comparison[:n_per_type]
    
    return [(q['question'], q.get('type', 'unknown')) for q in samples]


def load_2wiki_samples(n_total=10):
    """Load 2WikiMultihopQA samples with all 4 types"""
    with open('2WikiMultihopQA/2wikimultihopqa_sample_200.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Separate by type
    by_type = defaultdict(list)
    for q in data:
        qtype = q.get('type', 'unknown')
        by_type[qtype].append(q)
    
    # Sample from each type proportionally
    # compositional (45.5%), comparison (25%), bridge_comparison (19%), inference (10.5%)
    samples = []
    samples.extend(by_type['compositional'][:4])  # 4 compositional
    samples.extend(by_type['comparison'][:3])      # 3 comparison
    samples.extend(by_type['bridge_comparison'][:2])  # 2 bridge_comparison
    samples.extend(by_type['inference'][:1])       # 1 inference
    
    return [(q['question'], q.get('type', 'unknown')) for q in samples]


def load_musique_samples(n_total=10):
    """Load MuSiQue samples with varying hop counts"""
    with open('MuSiQue/musique_qa_sample_200.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Categorize by number of hops
    by_hops = defaultdict(list)
    for q in data:
        decomp = q.get('question_decomposition', [])
        n_hops = len(decomp) if decomp else 0
        by_hops[n_hops].append(q)
    
    # Sample from each hop count
    samples = []
    samples.extend(by_hops[2][:3])  # 3 questions with 2 hops
    samples.extend(by_hops[3][:4])  # 4 questions with 3 hops
    samples.extend(by_hops[4][:3])  # 3 questions with 4 hops
    
    return [(q['question'], f"{len(q.get('question_decomposition', []))}-hop") for q in samples]


async def test_decomposition(client: AsyncOpenAI, question: str, label: str, original_type: str):
    """Test decomposition on a single question"""
    result = await decompose_query(client, question)
    
    if result['success']:
        decomp = result['decomposition']
        
        # Collect statistics
        n_subqs = len(decomp.subquestions)
        independent = len([sq for sq in decomp.subquestions if not sq.depends_on])
        has_placeholders = any("[SQ" in sq.question for sq in decomp.subquestions)
        
        return {
            'label': label,
            'original_type': original_type,
            'predicted_type': decomp.question_type,
            'n_subquestions': n_subqs,
            'independent_subqs': independent,
            'has_placeholders': has_placeholders,
            'success': True,
            'question': question,
            'reasoning': decomp.reasoning,
            'subquestions': [
                {
                    'id': sq.id,
                    'question': sq.question,
                    'depends_on': sq.depends_on
                }
                for sq in decomp.subquestions
            ]
        }
    else:
        return {
            'label': label,
            'original_type': original_type,
            'success': False,
            'error': result.get('error', 'Unknown error'),
            'question': question
        }


async def main():
    """Test decomposition on 30 real questions from all datasets"""
    
    # Load environment
    load_dotenv()
    
    # Initialize OpenAI client
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    print("="*100)
    print("Comprehensive Query Decomposition Test - 30 Real Questions")
    print("="*100)
    print()
    
    # Load samples from each dataset
    print("Loading samples from datasets...")
    hotpotqa_samples = load_hotpotqa_samples(n_per_type=5)
    wiki2_samples = load_2wiki_samples(n_total=10)
    musique_samples = load_musique_samples(n_total=10)
    
    print(f"  HotpotQA: {len(hotpotqa_samples)} questions")
    print(f"  2WikiMultihopQA: {len(wiki2_samples)} questions")
    print(f"  MuSiQue: {len(musique_samples)} questions")
    print(f"  TOTAL: {len(hotpotqa_samples) + len(wiki2_samples) + len(musique_samples)} questions")
    print()
    
    # Run tests
    all_results = []
    
    # Test HotpotQA
    print("="*100)
    print("HotpotQA - 10 Questions")
    print("="*100)
    for idx, (question, qtype) in enumerate(hotpotqa_samples, 1):
        label = f"HotpotQA_{idx:02d}"
        print(f"\n[{label}] Type: {qtype}")
        try:
            print(f"Q: {question[:100]}..." if len(question) > 100 else f"Q: {question}")
        except UnicodeEncodeError:
            print(f"Q: {question.encode('utf-8', errors='replace').decode('utf-8')[:100]}...")
        
        result = await test_decomposition(client, question, label, qtype)
        all_results.append(result)
        
        if result['success']:
            print(f"   -> Predicted: {result['predicted_type']}, {result['n_subquestions']} sub-questions")
            # Print actual sub-questions
            for sq in result['subquestions']:
                deps = f" <- {', '.join(sq['depends_on'])}" if sq['depends_on'] else ""
                try:
                    print(f"      {sq['id']}: {sq['question']}{deps}")
                except UnicodeEncodeError:
                    print(f"      {sq['id']}: {sq['question'].encode('utf-8', errors='replace').decode('utf-8')}{deps}")
        else:
            print(f"   -> ERROR: {result['error']}")
        
        await asyncio.sleep(0.2)  # Rate limiting
    
    # Test 2WikiMultihopQA
    print("\n" + "="*100)
    print("2WikiMultihopQA - 10 Questions")
    print("="*100)
    for idx, (question, qtype) in enumerate(wiki2_samples, 1):
        label = f"2Wiki_{idx:02d}"
        print(f"\n[{label}] Type: {qtype}")
        try:
            print(f"Q: {question[:100]}..." if len(question) > 100 else f"Q: {question}")
        except UnicodeEncodeError:
            print(f"Q: {question.encode('utf-8', errors='replace').decode('utf-8')[:100]}...")
        
        result = await test_decomposition(client, question, label, qtype)
        all_results.append(result)
        
        if result['success']:
            print(f"   -> Predicted: {result['predicted_type']}, {result['n_subquestions']} sub-questions")
            # Print actual sub-questions
            for sq in result['subquestions']:
                deps = f" <- {', '.join(sq['depends_on'])}" if sq['depends_on'] else ""
                try:
                    print(f"      {sq['id']}: {sq['question']}{deps}")
                except UnicodeEncodeError:
                    print(f"      {sq['id']}: {sq['question'].encode('utf-8', errors='replace').decode('utf-8')}{deps}")
        else:
            print(f"   -> ERROR: {result['error']}")
        
        await asyncio.sleep(0.2)
    
    # Test MuSiQue
    print("\n" + "="*100)
    print("MuSiQue - 10 Questions")
    print("="*100)
    for idx, (question, qtype) in enumerate(musique_samples, 1):
        label = f"MuSiQue_{idx:02d}"
        print(f"\n[{label}] Hops: {qtype}")
        try:
            print(f"Q: {question[:100]}..." if len(question) > 100 else f"Q: {question}")
        except UnicodeEncodeError:
            print(f"Q: {question.encode('utf-8', errors='replace').decode('utf-8')[:100]}...")
        
        result = await test_decomposition(client, question, label, qtype)
        all_results.append(result)
        
        if result['success']:
            print(f"   -> Predicted: {result['predicted_type']}, {result['n_subquestions']} sub-questions")
            # Print actual sub-questions
            for sq in result['subquestions']:
                deps = f" <- {', '.join(sq['depends_on'])}" if sq['depends_on'] else ""
                try:
                    print(f"      {sq['id']}: {sq['question']}{deps}")
                except UnicodeEncodeError:
                    print(f"      {sq['id']}: {sq['question'].encode('utf-8', errors='replace').decode('utf-8')}{deps}")
        else:
            print(f"   -> ERROR: {result['error']}")
        
        await asyncio.sleep(0.2)
    
    # Generate summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"\nOverall Success Rate: {len(successful)}/{len(all_results)} ({len(successful)/len(all_results)*100:.1f}%)")
    
    # By dataset
    print("\nBy Dataset:")
    for dataset_name in ['HotpotQA', '2Wiki', 'MuSiQue']:
        dataset_results = [r for r in all_results if r['label'].startswith(dataset_name)]
        dataset_success = [r for r in dataset_results if r['success']]
        print(f"  {dataset_name}: {len(dataset_success)}/{len(dataset_results)} ({len(dataset_success)/len(dataset_results)*100:.1f}%)")
    
    # Type distribution (predicted)
    if successful:
        print("\nPredicted Type Distribution:")
        type_counts = defaultdict(int)
        for r in successful:
            type_counts[r['predicted_type']] += 1
        for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {qtype}: {count}")
    
    # Sub-question statistics
    if successful:
        print("\nSub-Question Statistics:")
        subq_counts = [r['n_subquestions'] for r in successful]
        print(f"  Average: {sum(subq_counts)/len(subq_counts):.2f}")
        print(f"  Range: {min(subq_counts)} - {max(subq_counts)}")
        print(f"  Distribution:")
        from collections import Counter
        for n, count in sorted(Counter(subq_counts).items()):
            print(f"    {n} sub-questions: {count} questions")
    
    # Placeholder usage
    if successful:
        with_placeholders = len([r for r in successful if r['has_placeholders']])
        print(f"\nPlaceholder Usage:")
        print(f"  Questions with [SQ{{N}}_Answer]: {with_placeholders}/{len(successful)} ({with_placeholders/len(successful)*100:.1f}%)")
    
    # Failed cases
    if failed:
        print(f"\nFailed Cases ({len(failed)}):")
        for r in failed:
            print(f"  [{r['label']}] {r['original_type']}: {r['error']}")
    
    # Save detailed results
    output_file = 'test_decomposition_30_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Print some example decompositions
    print("\n" + "="*100)
    print("EXAMPLE DECOMPOSITIONS (First 3 from each dataset)")
    print("="*100)
    
    for dataset_name in ['HotpotQA', '2Wiki', 'MuSiQue']:
        dataset_results = [r for r in successful if r['label'].startswith(dataset_name)][:3]
        
        for r in dataset_results:
            print(f"\n{'='*100}")
            print(f"[{r['label']}] Original Type: {r['original_type']}, Predicted: {r['predicted_type']}")
            print(f"{'='*100}")
            print(f"Question: {r['question']}")
            print(f"Reasoning: {r['reasoning']}")
            print(f"\nSub-questions ({len(r['subquestions'])}):")
            for sq in r['subquestions']:
                deps = f" (depends on: {', '.join(sq['depends_on'])})" if sq['depends_on'] else " (independent)"
                print(f"  {sq['id']}: {sq['question']}{deps}")
    
    return all_results


if __name__ == "__main__":
    results = asyncio.run(main())
