"""
Simple test: Check if entity extraction now returns multiple types
"""
import asyncio
import json
from hybrid_retrieval import initialize_llm_client
from Prompt.entity_extraction_prompt import ENTITY_EXTRACTION_PROMPT

async def test_extraction():
    client = initialize_llm_client()
    
    query = "What role do state institutions play in Argentina's education system?"
    
    print("="*80)
    print("SIMPLE TEST: Entity Extraction with Multiple Types")
    print("="*80)
    print(f"\nQuery: {query}\n")
    
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        print("Extraction Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Check if entities have possible_types
        if 'entities' in result:
            print("\n" + "="*80)
            print("ENTITY SUMMARY:")
            print("="*80)
            
            for i, entity in enumerate(result['entities'], 1):
                name = entity.get('entity_name', 'N/A')
                role = entity.get('role', 'N/A')
                importance = entity.get('importance', 'N/A')
                
                print(f"\n{i}. {name} ({role}, {importance})")
                
                # Check for possible_types
                possible_types = entity.get('possible_types', [])
                if possible_types:
                    print(f"   ✅ Has {len(possible_types)} possible types:")
                    for j, type_info in enumerate(possible_types, 1):
                        t = type_info.get('type', 'N/A')
                        st = type_info.get('subtype', 'N/A')
                        print(f"      {j}) {t}/{st}")
                else:
                    # Check old format
                    t = entity.get('type', 'N/A')
                    st = entity.get('subtype', 'N/A')
                    if t != 'N/A':
                        print(f"   ❌ Old format: {t}/{st}")
                    else:
                        print(f"   ❌ No type information")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_extraction())
