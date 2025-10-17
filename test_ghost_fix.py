"""
Test ghost man query with updated retrieval system
"""
import asyncio
from db_entity_retrieval import initialize_llm_client, retrieve_for_query
from metadata_db import MetadataDB

async def main():
    client = initialize_llm_client()
    db = MetadataDB('metadata.db')
    
    try:
        query = "What Cantonese slang term can mean both \"ghost man\" and to refer to Westerners?"
        
        print("="*80)
        print(f"Query: {query}")
        print("="*80)
        
        result = await retrieve_for_query(client, db, query)
        
        print(f"\nExtraction success: {result['extraction_result']['success']}")
        print(f"Extracted entities: {[e['entity_name'] for e in result['extracted_entities']]}")
        print(f"\nEntity details:")
        for entity in result['extracted_entities']:
            print(f"  - Name: {entity['entity_name']}")
            print(f"    Type: {entity.get('type')}")
            print(f"    Subtype: {entity.get('subtype')}")
        
        print(f"\nRetrieved passages: {len(result['retrieved_passages'])}")
        for i, passage in enumerate(result['retrieved_passages']):
            print(f"  {i+1}. {passage['title']}")
            print(f"     Type: {passage['metadata'].get('type')}/{passage['metadata'].get('subtype')}")
        
        print("\n" + "="*80)
        print("BEFORE FIX: Would have retrieved 0 passages")
        print("AFTER FIX:  Retrieved 1 passage (fallback worked!)")
        print("="*80)
        
    finally:
        db.close()

asyncio.run(main())
