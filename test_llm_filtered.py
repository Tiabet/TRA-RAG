"""
Test LLM-Filtered Retrieval System
===================================
Comprehensive testing with 2-stage filtering
"""
import asyncio
import json
from llm_filtered_retrieval import initialize_llm_client, retrieve_for_query
from metadata_db import MetadataDB


async def main():
    client = initialize_llm_client()
    db = MetadataDB('metadata_v2.db')
    
    try:
        # Load test queries
        with open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Select test cases (first 20 for quick test)
        num_tests = 20
        
        print("="*80)
        print(f"LLM-FILTERED RETRIEVAL TEST: {num_tests} queries")
        print("="*80)
        print("2-Stage Filtering:")
        print("  Stage 1: Type/Subtype matching")
        print("  Stage 2: LLM semantic filtering")
        print("="*80)
        
        # Statistics
        stats = {
            'total': 0,
            'extraction_success': 0,
            'extraction_failed': 0,
            'retrieved': 0,
            'no_match': 0,
            'stage1_total': 0,
            'stage2_filtered': 0,
            'llm_filter_errors': 0
        }
        
        # Question type statistics
        type_stats = {
            'bridge': {'total': 0, 'retrieved': 0, 'no_match': 0},
            'comparison': {'total': 0, 'retrieved': 0, 'no_match': 0}
        }
        
        unmatched_queries = []
        
        print("\nProcessing queries...")
        print("-" * 80)
        
        for i, item in enumerate(test_data[:num_tests]):
            query = item['question']
            q_type = item.get('type', 'unknown')
            
            print(f"\n[{i+1}/{num_tests}] {query[:70]}...")
            
            stats['total'] += 1
            if q_type in type_stats:
                type_stats[q_type]['total'] += 1
            
            # Retrieve with 2-stage filtering
            result = await retrieve_for_query(client, db, query)
            
            # Check extraction
            if not result['extraction_result']['success']:
                stats['extraction_failed'] += 1
                unmatched_queries.append({
                    'index': i + 1,
                    'type': q_type,
                    'question': query,
                    'reason': 'extraction_failed'
                })
                print(f"  [FAIL] Extraction failed")
                if q_type in type_stats:
                    type_stats[q_type]['no_match'] += 1
                continue
            
            stats['extraction_success'] += 1
            
            # Show entities
            entities = [e['entity_name'] for e in result['extracted_entities']]
            print(f"  Entities: {entities}")
            
            # Show 2-stage filtering process
            for entity_result in result['retrieval_info']['entity_results']:
                stage1 = entity_result['stage1_candidates']
                stage2_filtered = entity_result['stage2_filtered']
                final = entity_result['final_passages']
                
                stats['stage1_total'] += stage1
                stats['stage2_filtered'] += stage2_filtered
                
                print(f"    '{entity_result['entity_name']}':")
                print(f"      Stage 1: {stage1} candidates")
                print(f"      Stage 2: -{stage2_filtered} filtered → {final} final")
                
                if 'llm_filter_error' in entity_result:
                    stats['llm_filter_errors'] += 1
                    print(f"      ⚠️ LLM filter error: {entity_result['llm_filter_error'][:50]}...")
                
                # Show LLM reasoning (first passage only)
                if 'llm_reasoning' in entity_result and entity_result['llm_reasoning']:
                    reasoning = entity_result['llm_reasoning'][0]
                    print(f"      LLM: '{reasoning['title']}' ({reasoning['confidence']})")
                    print(f"           {reasoning['reasoning'][:60]}...")
            
            # Check retrieval
            num_passages = len(result['retrieved_passages'])
            
            if num_passages == 0:
                stats['no_match'] += 1
                unmatched_queries.append({
                    'index': i + 1,
                    'type': q_type,
                    'question': query,
                    'reason': 'no_passages',
                    'entities': entities
                })
                print(f"  [X] No passages retrieved")
                if q_type in type_stats:
                    type_stats[q_type]['no_match'] += 1
            else:
                stats['retrieved'] += 1
                print(f"  [OK] {num_passages} passages:")
                for p in result['retrieved_passages'][:3]:
                    print(f"       - {p['title']}")
                if q_type in type_stats:
                    type_stats[q_type]['retrieved'] += 1
        
        # Print summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"Total queries: {stats['total']}")
        
        print(f"\nEntity Extraction:")
        print(f"  Success: {stats['extraction_success']} ({stats['extraction_success']/stats['total']*100:.1f}%)")
        print(f"  Failed:  {stats['extraction_failed']} ({stats['extraction_failed']/stats['total']*100:.1f}%)")
        
        print(f"\nRetrieval Results:")
        print(f"  Retrieved:  {stats['retrieved']} ({stats['retrieved']/stats['total']*100:.1f}%)")
        print(f"  No match:   {stats['no_match']} ({stats['no_match']/stats['total']*100:.1f}%)")
        
        print(f"\n2-Stage Filtering Stats:")
        print(f"  Stage 1 (Type filter): {stats['stage1_total']} total candidates")
        print(f"  Stage 2 (LLM filter):  -{stats['stage2_filtered']} filtered out")
        if stats['stage1_total'] > 0:
            print(f"  Filter rate: {stats['stage2_filtered']/stats['stage1_total']*100:.1f}% filtered by LLM")
        print(f"  LLM errors: {stats['llm_filter_errors']}")
        
        # Question type analysis
        print("\n" + "="*80)
        print("QUESTION TYPE ANALYSIS")
        print("="*80)
        
        for q_type in ['bridge', 'comparison']:
            if type_stats[q_type]['total'] == 0:
                continue
            ts = type_stats[q_type]
            print(f"\n{q_type.upper()}:")
            print(f"  Total: {ts['total']}")
            print(f"  Retrieved: {ts['retrieved']} ({ts['retrieved']/ts['total']*100:.1f}%)")
            print(f"  No match:  {ts['no_match']} ({ts['no_match']/ts['total']*100:.1f}%)")
        
        # Unmatched queries
        if unmatched_queries:
            print("\n" + "="*80)
            print(f"UNMATCHED QUERIES: {len(unmatched_queries)}")
            print("="*80)
            for uq in unmatched_queries[:5]:
                print(f"\n[{uq['index']}] {uq['question'][:70]}...")
                print(f"    Reason: {uq['reason']}")
                if 'entities' in uq:
                    print(f"    Entities: {uq['entities']}")
        
        # Save results
        output_file = f'test_llm_filtered_{num_tests}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'num_tests': num_tests,
                'stats': stats,
                'type_stats': type_stats,
                'unmatched_queries': unmatched_queries
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print("="*80)
        
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
