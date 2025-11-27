"""
Test Leonard Logsdail question with full metadata (no truncation)
"""
import asyncio
from openai import AsyncOpenAI
from hybrid_retrieval import retrieve_for_entity_hybrid
from metadata_db import MetadataDB

async def test_leonard_full_metadata():
    """Test if full metadata (including relations) is now visible to LLM"""
    
    client = AsyncOpenAI(
        api_key="934c4b67-75e8-469f-ba00-1c4e0037ecf0",
        base_url="https://api.chatanywhere.tech"
    )
    
    db = MetadataDB('metadata_v2.db')
    
    print("\n" + "="*80)
    print("Testing Leonard Logsdail with FULL metadata (no truncation)")
    print("="*80)
    
    query = "Leonard Logsdail had a cameo role in the biographical film directed by whom?"
    
    entity = {
        "entity_name": "Leonard Logsdail",
        "possible_types": [
            {"type": "Person", "subtype": "Artist"},
            {"type": "Person", "subtype": "Actor"},
            {"type": "Person", "subtype": "PublicFigure"}
        ]
    }
    
    print(f"\nQuery: {query}")
    print(f"Entity: {entity}")
    
    # Run hybrid search
    print("\n" + "-"*80)
    print("Running retrieve_for_entity_hybrid...")
    print("-"*80)
    
    results, stats = await retrieve_for_entity_hybrid(
        client=client,
        db=db,
        query=query,
        entity=entity,
        use_fts=True,
        apply_llm_filter_stage1a=True
    )
    
    print(f"\n✓ Retrieved {len(results)} passages")
    print(f"\nStats: {stats}")
    
    print("\n" + "-"*80)
    print("Retrieved Passages:")
    print("-"*80)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['title']}")
        metadata = result.get('metadata', {})
        
        # Check if relations exist
        if 'relations' in metadata:
            print(f"  ✓ HAS RELATIONS:")
            for rel in metadata['relations'][:2]:  # Show first 2 relations
                print(f"    - {rel.get('relation', 'N/A')}: {rel.get('target', 'N/A')}")
        else:
            print(f"  ✗ NO RELATIONS")
        
        if 'appeared_in' in metadata.get('relations', [{}])[0]:
            print(f"  ✓✓✓ FOUND 'appeared_in' relation!")
    
    print("\n" + "="*80)
    if len(results) > 0:
        print("✓ SUCCESS: LLM kept Leonard Logsdail passage!")
        print("Check if 'relations' field with 'appeared_in' was visible to LLM")
    else:
        print("✗ FAIL: LLM still filtered out Leonard Logsdail")
        print("Relations field may still not be visible")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_leonard_full_metadata())
