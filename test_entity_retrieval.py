"""
Test Entity-based Retrieval System
===================================
Tests the retrieval system with real-time entity extraction from queries.
"""
import json
import asyncio
from entity_based_retrieval import retrieve_for_query, initialize_llm_client


def load_data():
    """Load metadata and test queries"""
    # Load pure metadata
    with open('HotpotQA/hotpotqa_sample_200_pure_metadata.json', 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    
    # Load original dataset for test queries
    with open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8') as f:
        query_data = json.load(f)
    
    return metadata_list, query_data


async def test_retrieval(client, metadata_list, query_data, num_tests=10):
    """Test retrieval on multiple queries with real-time entity extraction"""
    
    print("="*80)
    print("ENTITY-BASED RETRIEVAL TEST (Real-time Entity Extraction)")
    print("="*80)
    
    success_count = 0
    no_match_count = 0
    multi_match_count = 0
    extraction_failures = 0
    
    results_summary = []
    
    for i, item in enumerate(query_data[:num_tests]):
        print(f"\n{'='*80}")
        print(f"Test {i+1}/{num_tests}")
        print(f"{'='*80}")
        
        query = item['question']
        question_type = item.get('type', 'unknown')
        
        print(f"Question: {query[:100]}...")
        print(f"Type: {question_type}")
        
        # Perform real-time entity extraction and retrieval
        result = await retrieve_for_query(client, metadata_list, query)
        
        extraction_success = result['extraction_result']['success']
        extracted_entities = result['extracted_entities']
        
        if not extraction_success:
            print(f"[FAIL] Entity extraction failed: {result['extraction_result'].get('error')}")
            extraction_failures += 1
            continue
        
        print(f"Extracted entities ({len(extracted_entities)}):")
        for entity in extracted_entities:
            print(f"  - {entity['entity_name']} ({entity['type']}/{entity.get('subtype', 'N/A')})")
        
        num_retrieved = len(result['retrieved_passages'])
        print(f"\nRetrieved: {num_retrieved} passages")
        
        # Show retrieval status for each entity
        for entity_result in result['retrieval_info']['entity_results']:
            status_icon = "[OK]" if entity_result['matches_found'] > 0 else "[X]"
            print(f"  {status_icon} {entity_result['entity_name']}: {entity_result['matches_found']} matches")
        
        # Show retrieved passages
        if num_retrieved > 0:
            print(f"\nRetrieved passages:")
            for j, passage in enumerate(result['retrieved_passages'][:3]):  # Show max 3
                print(f"  {j+1}. {passage['title']}")
        
        # Track statistics
        if num_retrieved == 0:
            no_match_count += 1
            status = "NO_MATCH"
        elif num_retrieved == 1:
            success_count += 1
            status = "SINGLE_MATCH"
        else:
            multi_match_count += 1
            status = "MULTI_MATCH"
        
        results_summary.append({
            'query': query[:80],
            'type': question_type,
            'num_entities': len(extracted_entities),
            'num_retrieved': num_retrieved,
            'status': status
        })
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total tests: {num_tests}")
    print(f"Extraction failures: {extraction_failures}")
    print(f"Single match: {success_count} ({success_count/num_tests*100:.1f}%)")
    print(f"Multiple matches: {multi_match_count} ({multi_match_count/num_tests*100:.1f}%)")
    print(f"No matches: {no_match_count} ({no_match_count/num_tests*100:.1f}%)")
    
    return results_summary


async def test_custom_queries(client, metadata_list):
    """Test with custom queries"""
    print("\n" + "="*80)
    print("CUSTOM QUERY TESTS (Real-time)")
    print("="*80)
    
    custom_queries = [
        "Who is the founder of Tesla?",
        "Which city is larger, Tokyo or Seoul?",
        "What year did World War II end?",
    ]
    
    for i, query in enumerate(custom_queries, 1):
        print(f"\n{i}. Query: {query}")
        result = await retrieve_for_query(client, metadata_list, query)
        
        if result['extraction_result']['success']:
            entities = [e['entity_name'] for e in result['extracted_entities']]
            print(f"   Entities: {entities}")
            print(f"   Retrieved: {len(result['retrieved_passages'])} passages")
            for j, p in enumerate(result['retrieved_passages'][:2]):
                print(f"     - {p['title']}")
        else:
            print(f"   [FAIL] Extraction failed")


async def main():
    # Initialize LLM client
    print("Initializing LLM client...")
    client = initialize_llm_client()
    print("OK - LLM client initialized")
    
    # Load data
    print("\nLoading data...")
    metadata_list, query_data = load_data()
    print(f"OK - Loaded {len(metadata_list)} metadata entries")
    print(f"OK - Loaded {len(query_data)} test queries")
    
    # Run main tests
    results = await test_retrieval(client, metadata_list, query_data, num_tests=10)
    
    # Test custom queries
    await test_custom_queries(client, metadata_list)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
