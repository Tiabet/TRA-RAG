"""Test SQ3 Entity Extraction - Raw Output"""
import asyncio
import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from Prompt.subquestion_entity_extraction_prompt import SUBQUESTION_ENTITY_EXTRACTION_PROMPT

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
    
    # Format prompt
    formatted_prompt = SUBQUESTION_ENTITY_EXTRACTION_PROMPT.replace(
        "__SUBQUESTION__", 
        subquestion
    )
    
    if previous_context:
        formatted_prompt = formatted_prompt.replace(
            "{{previous_context}}", 
            previous_context
        )
    else:
        formatted_prompt = formatted_prompt.replace(
            "{{previous_context}}", 
            "(None - this is the first sub-question)"
        )
    
    # Call LLM
    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": formatted_prompt}
        ],
        temperature=0.1,
        max_tokens=1024
    )
    
    result_text = response.choices[0].message.content.strip()
    
    print("=" * 80)
    print("RAW LLM OUTPUT:")
    print("=" * 80)
    print(result_text)
    print("\n" + "=" * 80)
    
    # Try to parse
    if result_text.startswith('```json'):
        result_text = result_text[7:]
    if result_text.startswith('```'):
        result_text = result_text[3:]
    if result_text.endswith('```'):
        result_text = result_text[:-3]
    result_text = result_text.strip()
    
    try:
        result = json.loads(result_text)
        print("PARSED JSON:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"PARSE ERROR: {e}")

asyncio.run(test())
