"""
Test Multi-Dataset Query Decomposition
========================================
Tests the updated type-agnostic decomposition on all 3 datasets.

Tests:
1. HotpotQA: Bridge + Comparison (baseline)
2. 2WikiMultihopQA: Compositional, inference, bridge_comparison
3. MuSiQue: Long chains (3-4 hops), #N placeholders
"""

import asyncio
import json
from openai import AsyncOpenAI
from query_decomposition import decompose_query


# Test questions from each dataset
TEST_QUESTIONS = {
    "HotpotQA_bridge": "Where was the director of film Doctor Krishna born?",
    "HotpotQA_comparison": "Are both Stephen R. Donaldson and Michael Moorcock science fiction writers?",
    
    "2Wiki_compositional": "What body of water is by the headquarters location of Wipac?",
    "2Wiki_inference": "Who is the father-in-law of Elizabeth Somerset, Baroness Herbert?",
    "2Wiki_bridge_comparison": "Which film has the director who died earlier, A Doctor's Diary or Wild Rovers?",
    
    "MuSiQue_3hop": "What is the symbol of the Saints from the city where the headquarters of the manufacturer of McAfee's Benchmark called?",
    "MuSiQue_4hop": "In which county was the performer of Put a Little Love in Your Heart born?"
}


async def test_decomposition(client: AsyncOpenAI, question: str, label: str):
    """Test decomposition on a single question"""
    print(f"\n{'='*80}")
    print(f"Test: {label}")
    print(f"{'='*80}")
    print(f"Question: {question}\n")
    
    result = await decompose_query(client, question)
    
    if result['success']:
        decomp = result['decomposition']
        print(f"Type: {decomp.question_type}")
        print(f"Reasoning: {decomp.reasoning}")
        print(f"\nSub-questions ({len(decomp.subquestions)}):")
        
        for sq in decomp.subquestions:
            deps = f" (depends on: {', '.join(sq.depends_on)})" if sq.depends_on else " (independent)"
            print(f"  {sq.id}: {sq.question}{deps}")
            print(f"       → {sq.reasoning}")
        
        # Test placeholder formats
        print(f"\nPlaceholder Check:")
        for sq in decomp.subquestions:
            if "[SQ" in sq.question:
                print(f"  {sq.id}: Uses [SQ{{N}}_Answer] format OK")
            elif "#" in sq.question and any(c.isdigit() for c in sq.question):
                print(f"  {sq.id}: Uses #{{N}} format OK")
        
        return True
    else:
        print(f"ERROR: {result['error']}")
        return False


async def main():
    """Test decomposition on all datasets"""
    
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize OpenAI client
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    print("="*80)
    print("Multi-Dataset Query Decomposition Test")
    print("="*80)
    print(f"Testing {len(TEST_QUESTIONS)} questions across 3 datasets")
    print("HotpotQA: 2 questions (bridge, comparison)")
    print("2WikiMultihopQA: 3 questions (compositional, inference, bridge_comparison)")
    print("MuSiQue: 2 questions (3-hop, 4-hop)")
    
    results = {}
    for label, question in TEST_QUESTIONS.items():
        success = await test_decomposition(client, question, label)
        results[label] = success
        await asyncio.sleep(1)  # Rate limiting
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"Successful decompositions: {success_count}/{total_count}")
    
    print("\nBy dataset:")
    hotpotqa = sum(1 for k, v in results.items() if k.startswith("HotpotQA") and v)
    wiki2 = sum(1 for k, v in results.items() if k.startswith("2Wiki") and v)
    musique = sum(1 for k, v in results.items() if k.startswith("MuSiQue") and v)
    
    print(f"  HotpotQA: {hotpotqa}/2")
    print(f"  2WikiMultihopQA: {wiki2}/3")
    print(f"  MuSiQue: {musique}/2")
    
    if success_count == total_count:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[WARNING] {total_count - success_count} test(s) failed")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
