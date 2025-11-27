"""
Test Simple Reasoning Cases
============================
Test if LLM can perform simple reasoning like temporal calculations.
"""
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from metadata_db import MetadataDB
from query_decomposition import decompose_query
from sequential_answering import answer_subquestions_sequential, synthesize_final_answer

load_dotenv()


async def test_reasoning_case(client, db, main_query, gold_answer):
    """Test a single question."""
    
    print("=" * 80)
    print(f"Main Query: {main_query}")
    print(f"Gold Answer: {gold_answer}")
    print("=" * 80)
    
    # Decompose
    decomposition_result = await decompose_query(client, main_query)
    
    if not decomposition_result['success']:
        print(f"❌ Decomposition failed")
        return False
    
    decomposition = decomposition_result['decomposition']
    
    # Answer sub-questions
    answer_result = await answer_subquestions_sequential(
        client, db, decomposition,
        use_fts=True,
        apply_llm_filter_stage1a=True,
        verbose=False
    )
    
    if not answer_result['success']:
        print(f"❌ Answering failed")
        return False
    
    # Final synthesis
    final_answer = await synthesize_final_answer(client, decomposition)
    
    print(f"\n🎯 Predicted Answer: {final_answer}")
    print(f"✅ Gold Answer: {gold_answer}")
    
    # Check correctness
    gold_normalized = gold_answer.lower().strip()
    pred_normalized = final_answer.lower().strip()
    
    if gold_normalized in pred_normalized or pred_normalized in gold_normalized:
        print("✅ SUCCESS!")
        return True
    else:
        print("❌ FAILED")
        print("\nSub-question details:")
        for sq in decomposition.subquestions:
            print(f"  {sq.id}: {sq.question}")
            print(f"  Answer: {sq.answer}")
        return False


async def test_all_reasoning_cases():
    """Test multiple cases that require simple reasoning."""
    
    # Initialize
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    db = MetadataDB('metadata_v2.db')
    
    test_cases = [
        # Temporal reasoning
        {
            "query": "Seven years before the opening of the Brewer Fieldhouse in Columbia, Missouri, what was a campus of the University of Missouri known as?",
            "gold": "University Farm"
        },
        # Simple entity extraction (baseline)
        {
            "query": "Who proposed plan in which education in state institutions of Argentina is free at the initial, primary, secondary and tertiary levels and in the undergraduate university level?",
            "gold": "Dr. Alberto Taquini"
        },
    ]
    
    print("\n" + "=" * 80)
    print("Testing Simple Reasoning Cases")
    print("=" * 80 + "\n")
    
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}/{len(test_cases)}")
        print(f"{'=' * 80}")
        
        success = await test_reasoning_case(
            client, db,
            case["query"],
            case["gold"]
        )
        results.append(success)
        
        print()
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Success Rate: {sum(results)/len(results)*100:.1f}%")
    
    db.close()


if __name__ == "__main__":
    asyncio.run(test_all_reasoning_cases())
