"""
Quick test: First 5 queries with hybrid retrieval
"""
import asyncio
import json
from hybrid_retrieval import initialize_llm_client, retrieve_for_query
from metadata_db import MetadataDB


async def main():
    client = initialize_llm_client()
    db = MetadataDB('metadata_v2.db')
    
    try:
        # Load test queries
        with open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        num_tests = 5
        
        print("="*80)
        print(f"QUICK TEST: {num_tests} queries")
        print("="*80)
        
        for i, item in enumerate(test_data[:num_tests]):
            query = item['question']
            
            print(f"\n[{i+1}/{num_tests}] {query[:70]}...")
            
            result = await retrieve_for_query(client, db, query, use_fts=True)
            
            if not result['extraction_result']['success']:
                print(f"  [FAIL] Extraction failed")
                continue
            
            # Show entities
            entities = result['extracted_entities']
            print(f"  Entities: {len(entities)}")
            for e in entities:
                name = e['entity_name']
                num_types = len(e.get('possible_types', []))
                print(f"    - {name} ({num_types} types)")
            
            # Show results
            passages = result['retrieved_passages']
            print(f"  Retrieved: {len(passages)} passages")
            
            if passages:
                print(f"  Top 3:")
                for p in passages[:3]:
                    print(f"    - {p['title']}")
            
        print("\n" + "="*80)
        print("DONE")
        print("="*80)
        
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
