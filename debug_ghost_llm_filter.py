"""
Debug: Ghost man LLM filtering issue
"""
import asyncio
from llm_filtered_retrieval import (
    initialize_llm_client, 
    retrieve_passages_for_entity_stage1,
    llm_filter_candidates
)
from metadata_db import MetadataDB

async def main():
    client = initialize_llm_client()
    db = MetadataDB('metadata_v2.db')
    
    try:
        query = "What Cantonese slang term can mean both 'ghost man' and to refer to Westerners?"
        entity_name = "ghost man"
        entity_type = "Concept"
        entity_subtype = "SocialSystem"
        
        print("="*80)
        print("Debugging Ghost Man LLM Filtering")
        print("="*80)
        
        # Stage 1: Get candidates
        print("\nStage 1: Type/Subtype filtering")
        candidates = retrieve_passages_for_entity_stage1(
            db, entity_name, entity_type, entity_subtype, use_fts=True
        )
        print(f"Found {len(candidates)} candidates:")
        for c in candidates:
            print(f"  - {c['title']} ({c['metadata'].get('type')}/{c['metadata'].get('subtype')})")
            # Show metadata snippet
            if 'attributes' in c['metadata']:
                print(f"    Attributes: {list(c['metadata']['attributes'].keys())[:5]}")
        
        # Stage 2: LLM filtering
        print(f"\nStage 2: LLM semantic filtering")
        print(f"Query: {query}")
        print(f"Entity: {entity_name} ({entity_type}/{entity_subtype})")
        
        filter_result = await llm_filter_candidates(
            client, query, entity_name, entity_type, entity_subtype, candidates
        )
        
        print(f"\nLLM Filter Result:")
        print(f"  Success: {filter_result.get('success', True)}")
        
        if 'error' in filter_result:
            print(f"  Error: {filter_result['error']}")
        
        print(f"\n  Relevant passages: {len(filter_result.get('relevant_passages', []))}")
        for p in filter_result.get('relevant_passages', []):
            print(f"    ✓ {p['title']} ({p['confidence']})")
            print(f"      Reasoning: {p['reasoning']}")
        
        print(f"\n  Filtered out: {len(filter_result.get('filtered_out', []))}")
        for p in filter_result.get('filtered_out', []):
            print(f"    ✗ {p['title']}")
            print(f"      Reasoning: {p['reasoning']}")
        
        # Check if "Gweilo" exists in DB
        print("\n" + "="*80)
        print("Checking for 'Gweilo' in database")
        print("="*80)
        
        gweilo_candidates = db.search_by_entity_fts("Gweilo")
        print(f"Found {len(gweilo_candidates)} results for 'Gweilo':")
        for c in gweilo_candidates:
            print(f"  - {c['title']} ({c['metadata'].get('type')}/{c['metadata'].get('subtype')})")
            if 'attributes' in c['metadata']:
                attrs = c['metadata']['attributes']
                if 'description' in attrs:
                    print(f"    Description: {attrs['description'][:100]}...")
        
    finally:
        db.close()

asyncio.run(main())
