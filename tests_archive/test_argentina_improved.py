"""
Test Argentina case with improved prompts
==========================================
Check if the enhanced prompts help LLM find answers in passages.
"""
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from metadata_db import MetadataDB
from query_decomposition import decompose_query
from sequential_answering import answer_subquestions_sequential, synthesize_final_answer

load_dotenv()


async def test_argentina_improved():
    """Test Argentina case with improved answer extraction."""
    
    # Initialize
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    db = MetadataDB('metadata_v2.db')
    
    # Main query
    main_query = "Who proposed plan in which education in state institutions of Argentina is free at the initial, primary, secondary and tertiary levels and in the undergraduate university level?"
    
    print("=" * 80)
    print("Testing Argentina Case with Improved Prompts")
    print("=" * 80)
    print(f"Main Query: {main_query}\n")
    
    # Step 1: Decompose
    print("Step 1: Query Decomposition...")
    decomposition_result = await decompose_query(client, main_query)
    
    if not decomposition_result['success']:
        print(f"❌ Decomposition failed: {decomposition_result.get('error')}")
        return
    
    decomposition = decomposition_result['decomposition']
    print(f"✅ Decomposition successful")
    print(f"   Sub-questions: {len(decomposition.subquestions)}\n")
    
    for sq in decomposition.subquestions:
        print(f"   {sq.id}: {sq.question}")
    print()
    
    # Step 2: Answer sub-questions
    print("Step 2: Answering Sub-Questions...")
    print("-" * 80)
    
    answer_result = await answer_subquestions_sequential(
        client, db, decomposition,
        use_fts=True,
        apply_llm_filter_stage1a=True,
        verbose=True
    )
    
    if not answer_result['success']:
        print(f"❌ Answering failed: {answer_result.get('error')}")
        return
    
    print("\n" + "=" * 80)
    print("Sub-Question Results:")
    print("=" * 80)
    
    for sq in decomposition.subquestions:
        print(f"\n{sq.id}: {sq.question}")
        print(f"Answer: {sq.answer}")
        
        if hasattr(sq, 'retrieved_passages') and sq.retrieved_passages:
            print(f"Retrieved passages ({len(sq.retrieved_passages)}):")
            for p in sq.retrieved_passages[:5]:
                print(f"  - {p.get('title', 'Unknown')}")
    
    # Step 3: Final synthesis
    print("\n" + "=" * 80)
    print("Step 3: Final Answer Synthesis")
    print("=" * 80)
    
    final_answer = await synthesize_final_answer(client, decomposition)
    
    print(f"\n🎯 Final Answer: {final_answer}")
    print(f"✅ Gold Answer: Dr. Alberto Taquini")
    
    # Check if correct
    if "Taquini" in final_answer or "Alberto" in final_answer:
        print("\n✅ SUCCESS! Found the correct answer!")
    else:
        print("\n❌ FAILED: Answer not found")
    
    db.close()


if __name__ == "__main__":
    asyncio.run(test_argentina_improved())
