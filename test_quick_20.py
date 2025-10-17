"""
Quick test: First 20 queries with type fallback
"""
import asyncio
import json
from db_entity_retrieval import initialize_llm_client, retrieve_for_query
from metadata_db import MetadataDB

async def main():
    client = initialize_llm_client()
    db = MetadataDB('metadata.db')
    
    try:
        with open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        num_tests = 20
        results_summary = {
            'extraction_success': 0,
            'extraction_failed': 0,
            'single_match': 0,
            'multi_match': 0,
            'no_match': 0
        }
        
        print("="*80)
        print(f"Testing first {num_tests} queries with TYPE FALLBACK")
        print("="*80)
        
        for i, item in enumerate(test_data[:num_tests]):
            query = item['question']
            print(f"\n[{i+1}/{num_tests}] Query: {query[:70]}...")
            
            result = await retrieve_for_query(client, db, query)
            
            if result['extraction_result']['success']:
                results_summary['extraction_success'] += 1
                entities = [e['entity_name'] for e in result['extracted_entities']]
                print(f"  Entities: {entities}")
                
                num_passages = len(result['retrieved_passages'])
                if num_passages == 0:
                    results_summary['no_match'] += 1
                    print(f"  [X] No passages")
                elif num_passages == 1:
                    results_summary['single_match'] += 1
                    print(f"  [OK] 1 passage: {result['retrieved_passages'][0]['title']}")
                else:
                    results_summary['multi_match'] += 1
                    print(f"  [OK] {num_passages} passages")
            else:
                results_summary['extraction_failed'] += 1
                print(f"  [FAIL] Extraction failed")
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"Total: {num_tests}")
        print(f"\nExtraction:")
        print(f"  Success: {results_summary['extraction_success']} ({results_summary['extraction_success']/num_tests*100:.1f}%)")
        print(f"  Failed:  {results_summary['extraction_failed']} ({results_summary['extraction_failed']/num_tests*100:.1f}%)")
        
        print(f"\nRetrieval:")
        print(f"  Single:  {results_summary['single_match']} ({results_summary['single_match']/num_tests*100:.1f}%)")
        print(f"  Multi:   {results_summary['multi_match']} ({results_summary['multi_match']/num_tests*100:.1f}%)")
        print(f"  No match: {results_summary['no_match']} ({results_summary['no_match']/num_tests*100:.1f}%)")
        
        total_retrieved = results_summary['single_match'] + results_summary['multi_match']
        print(f"\n>>> Total retrieved: {total_retrieved}/{num_tests} ({total_retrieved/num_tests*100:.1f}%)")
        
    finally:
        db.close()

asyncio.run(main())
