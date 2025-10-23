"""
Test if transitive dependency fix resolves the SQ3 zero-passage problem.
Uses the actual question from results and performs full LLM-based decomposition.
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from metadata_db import MetadataDB
from query_decomposition import decompose_query
from sequential_answering import answer_subquestions_sequential, synthesize_final_answer
from llm_logger import init_logger, finalize_log

load_dotenv()

# Initialize LLM logger
logger = init_logger()

# Initialize
client = AsyncOpenAI(
    api_key=os.getenv('ALICE_OPENAI_KEY'),
    base_url=os.getenv('ALICE_CHAT_URL')
)
db = MetadataDB('metadata_v2.db')

async def test_transitive_dependency():
    """Test the specific case that was failing with full LLM decomposition."""
    
    # Use the EXACT question from the results
    main_query = "Seven years before the opening of the Brewer Fieldhouse in Columbia, Missouri, where was Chester Brewer working as head football coach and head basketball coach?"
    expected_answer = "University Farm"
    
    print("=" * 80)
    print("Testing Transitive Dependency Fix with Full LLM Decomposition")
    print("=" * 80)
    print(f"Main Query: {main_query}")
    print(f"Expected Answer: {expected_answer}")
    print()
    
    # Step 1: Decompose query using LLM
    print("Step 1: Query Decomposition (LLM)")
    print("-" * 80)
    decomposition_result = await decompose_query(client, main_query)
    
    if not decomposition_result['success']:
        print(f"❌ Decomposition failed: {decomposition_result.get('error', 'Unknown error')}")
        log_file = finalize_log()
        print(f"📄 LLM interactions logged to: {log_file}")
        return False
    
    decomposition = decomposition_result['decomposition']
    
    print(f"✅ Decomposed into {len(decomposition.subquestions)} sub-questions:")
    for sq in decomposition.subquestions:
        print(f"  {sq.id}: {sq.question}")
        if sq.depends_on:
            print(f"       → Depends on: {', '.join(sq.depends_on)}")
    print()
    
    # Step 2: Answer sub-questions sequentially
    print("Step 2: Sequential Answering")
    print("-" * 80)
    answer_result = await answer_subquestions_sequential(
        client, db, decomposition,
        use_fts=True,
        apply_llm_filter_stage1a=True,
        verbose=True
    )
    
    if not answer_result['success']:
        print(f"❌ Answering failed: {answer_result.get('error', 'Unknown error')}")
        log_file = finalize_log()
        print(f"📄 LLM interactions logged to: {log_file}")
        return False
    
    print("\n" + "=" * 80)
    print("Sub-Question Results")
    print("=" * 80)
    
    for sq in decomposition.subquestions:
        print(f"\n{sq.id}: {sq.question}")
        print(f"Answer: {sq.answer}")
        
        if hasattr(sq, 'retrieved_passages'):
            print(f"Retrieved Passages: {len(sq.retrieved_passages)}")
            for i, passage in enumerate(sq.retrieved_passages[:3], 1):
                print(f"  [{i}] {passage.get('title', 'Unknown')}")
    
    # Step 3: Final answer synthesis
    print("\n" + "=" * 80)
    print("Step 3: Final Answer Synthesis")
    print("-" * 80)
    final_answer = await synthesize_final_answer(client, decomposition)
    
    print("\n" + "=" * 80)
    print("Final Answer")
    print("=" * 80)
    print(f"Predicted: {final_answer}")
    print(f"Expected: {expected_answer}")
    
    # Check success
    final_answer_lower = final_answer.lower()
    if 'university farm' in final_answer_lower:
        print("\n✅ SUCCESS! Answer contains 'University Farm'")
        log_file = finalize_log()
        print(f"📄 LLM interactions logged to: {log_file}")
        return True
    else:
        print("\n❌ FAILED - Answer does not contain 'University Farm'")
        log_file = finalize_log()
        print(f"📄 LLM interactions logged to: {log_file}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_transitive_dependency())
    exit(0 if success else 1)
