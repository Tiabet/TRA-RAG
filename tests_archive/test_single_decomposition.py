"""
Single Question Decomposition Test
====================================
Tests query decomposition with detailed output.
"""

import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from query_decomposition import decompose_query

async def test_single_question():
    """Test decomposition on a single question"""
    
    # Load environment
    load_dotenv()
    
    # Initialize client
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    # Test question
    question = "Where was the director of film Doctor Krishna born?"
    
    print("="*100)
    print("Query Decomposition Test")
    print("="*100)
    print(f"Question: {question}")
    print()
    
    result = await decompose_query(client, question)
    
    print(f"Success: {result['success']}")
    print()
    
    if result['success']:
        decomp = result['decomposition']
        
        print(f"Question Type: {decomp.question_type}")
        print(f"Reasoning: {decomp.reasoning}")
        print(f"Number of Sub-Questions: {len(decomp.subquestions)}")
        print()
        
        print("Sub-Questions:")
        print("-" * 100)
        for sq in decomp.subquestions:
            deps = f" (depends on: {', '.join(sq.depends_on)})" if sq.depends_on else " (independent)"
            print(f"\n{sq.id}{deps}")
            print(f"  Question: {sq.question}")
            print(f"  Reasoning: {sq.reasoning}")
        
        print()
        print("="*100)
        print("✓ Test completed successfully!")
        
    else:
        print(f"✗ Error: {result.get('error', 'Unknown error')}")
        print()
        
        if 'raw_response' in result:
            print("Raw LLM Response:")
            print(result['raw_response'])

if __name__ == "__main__":
    asyncio.run(test_single_question())
