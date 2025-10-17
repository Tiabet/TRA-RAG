"""
Test Database-based Entity Retrieval
====================================
Comprehensive testing with real queries.
"""
import asyncio
import json
from db_entity_retrieval import initialize_llm_client, retrieve_for_query
from metadata_db import MetadataDB


async def main():
    # Initialize
    print("Initializing LLM client and database...")
    client = initialize_llm_client()
    db = MetadataDB('metadata.db')
    
    try:
        # Load test queries
        with open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"Loaded {len(test_data)} test queries\n")
        
        # Test on all queries in the file
        num_tests = len(test_data)
        results_summary = {
            'extraction_success': 0,
            'extraction_failed': 0,
            'single_match': 0,
            'multi_match': 0,
            'no_match': 0
        }
        
        print("="*80)
        print(f"Testing {num_tests} queries (full dataset)")
        print("="*80)
        
        for i, item in enumerate(test_data[:num_tests]):
            query = item['question']
            print(f"\n[{i+1}/{num_tests}] Query: {query[:80]}...")
            
            # Retrieve
            result = await retrieve_for_query(client, db, query)
            
            # Check extraction
            if result['extraction_result']['success']:
                results_summary['extraction_success'] += 1
                entities = [e['entity_name'] for e in result['extracted_entities']]
                print(f"  [OK] Entities: {entities}")
                
                # Check retrieval
                num_passages = len(result['retrieved_passages'])
                if num_passages == 0:
                    results_summary['no_match'] += 1
                    print(f"  [X] No passages retrieved")
                elif num_passages == 1:
                    results_summary['single_match'] += 1
                    print(f"  [OK] Single match: {result['retrieved_passages'][0]['title']}")
                else:
                    results_summary['multi_match'] += 1
                    print(f"  [OK] {num_passages} passages retrieved")
                    for j, p in enumerate(result['retrieved_passages'][:3]):
                        print(f"      {j+1}. {p['title']}")
            else:
                results_summary['extraction_failed'] += 1
                print(f"  [FAIL] Entity extraction failed: {result['extraction_result'].get('error', 'Unknown')}")
        
        # Summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"Total queries tested: {num_tests}")
        print(f"\nEntity Extraction:")
        print(f"  Success: {results_summary['extraction_success']} ({results_summary['extraction_success']/num_tests*100:.1f}%)")
        print(f"  Failed:  {results_summary['extraction_failed']} ({results_summary['extraction_failed']/num_tests*100:.1f}%)")
        
        print(f"\nRetrieval Results:")
        print(f"  Single match:  {results_summary['single_match']} ({results_summary['single_match']/num_tests*100:.1f}%)")
        print(f"  Multi match:   {results_summary['multi_match']} ({results_summary['multi_match']/num_tests*100:.1f}%)")
        print(f"  No match:      {results_summary['no_match']} ({results_summary['no_match']/num_tests*100:.1f}%)")
        
        # Special test: Search in nested values
        print("\n" + "="*80)
        print("SPECIAL TEST: Nested Value Search")
        print("="*80)
        print("\nSearching for 'Estonia' (should find in relations/attributes):")
        
        # Direct DB search
        results = db.search_by_entity("Estonia", search_title_only=False)
        print(f"Found {len(results)} passages with 'Estonia' in ANY value:")
        for i, r in enumerate(results[:5]):
            print(f"  {i+1}. {r['title']}")
            # Show where Estonia appears
            import json as js
            metadata_str = js.dumps(r['metadata'], ensure_ascii=False)
            if 'Estonia' in metadata_str:
                # Find the field
                for key, value in r['metadata'].items():
                    if isinstance(value, str) and 'Estonia' in value:
                        print(f"      → Found in: {key}")
                    elif isinstance(value, dict):
                        for k2, v2 in value.items():
                            if isinstance(v2, str) and 'Estonia' in v2:
                                print(f"      → Found in: {key}.{k2}")
        
        # FTS search
        print(f"\nUsing FTS:")
        results_fts = db.search_by_entity_fts("Estonia")
        print(f"Found {len(results_fts)} passages with FTS")
        
        # Performance comparison
        print("\n" + "="*80)
        print("PERFORMANCE TEST")
        print("="*80)
        
        import time
        test_entity = "Baltic Cup"
        
        # Title-only search
        start = time.perf_counter()
        results_title = db.search_by_entity(test_entity, search_title_only=True)
        time_title = (time.perf_counter() - start) * 1000
        
        # All-values search
        start = time.perf_counter()
        results_all = db.search_by_entity(test_entity, search_title_only=False)
        time_all = (time.perf_counter() - start) * 1000
        
        # FTS search
        start = time.perf_counter()
        results_fts = db.search_by_entity_fts(test_entity)
        time_fts = (time.perf_counter() - start) * 1000
        
        print(f"Search for '{test_entity}':")
        print(f"  Title-only:  {len(results_title)} results in {time_title:.2f}ms")
        print(f"  All-values:  {len(results_all)} results in {time_all:.2f}ms")
        print(f"  FTS:         {len(results_fts)} results in {time_fts:.2f}ms")
        
        if time_fts > 0:
            print(f"\n  FTS is {time_all/time_fts:.1f}x faster than all-values search")
        
        print("\n" + "="*80)
        print("✓ Testing complete!")
        print("="*80)
        
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
