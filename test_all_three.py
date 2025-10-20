"""
Test all 3 queries with multiple types system
"""
import asyncio
from hybrid_retrieval import initialize_llm_client, retrieve_for_query
from metadata_db import MetadataDB

async def test_query(client, db, query, expected_answers):
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    result = await retrieve_for_query(client, db, query, use_fts=True)
    
    # Show entities
    entities = result.get('extracted_entities', [])
    print(f"\nExtracted {len(entities)} entities:")
    for i, entity in enumerate(entities, 1):
        name = entity.get('entity_name')
        role = entity.get('role')
        importance = entity.get('importance')
        types_count = len(entity.get('possible_types', []))
        print(f"  {i}. {name} ({role}/{importance}) - {types_count} types")
    
    # Show retrieval summary
    retrieval_info = result.get('retrieval_info', {})
    entity_results = retrieval_info.get('entity_results', [])
    
    print(f"\nRetrieval summary:")
    for entity_info in entity_results:
        name = entity_info.get('entity_name')
        stage1b_info = entity_info.get('stage1b_type_info', {})
        type_candidates = stage1b_info.get('type_candidates', 0)
        llm_filtered = stage1b_info.get('llm_filtered', 0)
        final = entity_info.get('stage2_final', 0)
        
        print(f"  - {name}: {type_candidates} candidates → {llm_filtered} LLM → {final} final")
    
    # Check answers
    passages = result.get('retrieved_passages', [])
    print(f"\nTotal passages: {len(passages)}")
    
    if expected_answers:
        print(f"Expected answers:")
        for expected in expected_answers:
            found = any(expected.lower() in p.get('title', '').lower() for p in passages)
            status = "[FOUND]" if found else "[MISS]"
            print(f"  {status} {expected}")
    
    # Show top 3
    print(f"\nTop 3 results:")
    for i, p in enumerate(passages[:3], 1):
        title = p.get('title', 'N/A')
        t = p['metadata'].get('type', 'N/A')
        st = p['metadata'].get('subtype', 'N/A')
        print(f"  {i}. {title} ({t}/{st})")

async def main():
    client = initialize_llm_client()
    db = MetadataDB("metadata_v2.db")
    
    print("="*80)
    print("COMPREHENSIVE TEST: Multiple Types System")
    print("="*80)
    
    # Test 1: Stephen Graham
    await test_query(
        client, db,
        "What year was the actor who portrayed Stephen Graham in the 2006 film about the Nuremberg trials born?",
        ["Stephen Graham"]
    )
    
    # Test 2: Argentina education
    await test_query(
        client, db,
        "What role do state institutions play in Argentina's education system?",
        ["Education in Argentina", "Free education"]
    )
    
    # Test 3: Bee Cliff
    await test_query(
        client, db,
        "Where is Bee Cliff located?",
        ["Bee Cliff", "Watauga River"]
    )
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
