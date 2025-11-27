"""Test SQ3 Entity Extraction"""
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sequential_answering import extract_entities_from_subquestion

load_dotenv()

async def test():
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    subquestion = "What was the campus of the University of Missouri known as in 1922?"
    previous_context = """SQ1: What year did Brewer Fieldhouse open?
Answer: 1929

SQ2: Seven years before that?
Answer: 1922"""
    
    result = await extract_entities_from_subquestion(client, subquestion, previous_context)
    
    print("Success:", result['success'])
    print("\nExtracted Entities:")
    for ent in result.get('entities', []):
        print(f"\n  Entity: {ent.get('name', 'UNKNOWN')}")
        print(f"  Role: {ent.get('role', 'N/A')}")
        print(f"  Importance: {ent.get('importance', 'N/A')}")
        types = ent.get('types', [])
        if types:
            print(f"  Types:")
            for t in types[:2]:
                print(f"    - {t.get('type', 'Unknown')}/{t.get('subtype', 'Unknown')}")

asyncio.run(test())
