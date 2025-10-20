"""
Test Hybrid Retrieval System (Multiple Types)
==============================================
Testing with 20 queries to evaluate:
- Multiple type extraction (2-3 types per entity)
- Stage 1-A: Value matching (FTS)
- Stage 1-B: Type filtering (multiple types) + LLM filtering
- Stage 2: Merge results
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
        
        # Select test cases (first 20 for quick test)
        num_tests = 20
        
        print("="*80)
        print(f"HYBRID RETRIEVAL TEST (Multiple Types): {num_tests} queries")
        print("="*80)
        print("Pipeline:")
        print("  Stage 1-A: Value matching (FTS)")
        print("  Stage 1-B: Type filtering (2-3 types per entity) + LLM filtering")
        print("  Stage 2: Merge results")
        print("="*80)
        
        # Statistics
        stats = {
            'total': 0,
            'extraction_success': 0,
            'extraction_failed': 0,
            'retrieved': 0,
            'no_match': 0,
            'stage1a_total': 0,
            'stage1b_type_candidates': 0,
            'stage1b_llm_filtered': 0,
            'stage2_final': 0,
            'multiple_types_used': 0,
            'total_types_tried': 0
        }
        
        # Question type statistics
        type_stats = {
            'bridge': {'total': 0, 'retrieved': 0, 'no_match': 0},
            'comparison': {'total': 0, 'retrieved': 0, 'no_match': 0}
        }
        
        unmatched_queries = []
        type_coverage_examples = []  # Track examples of multiple type usage
        
        print("\nProcessing queries...")
        print("-" * 80)
        
        for i, item in enumerate(test_data[:num_tests]):
            query = item['question']
            q_type = item.get('type', 'unknown')
            
            print(f"\n[{i+1}/{num_tests}] {query[:70]}...")
            
            stats['total'] += 1
            if q_type in type_stats:
                type_stats[q_type]['total'] += 1
            
            # Retrieve with hybrid approach
            result = await retrieve_for_query(client, db, query, use_fts=True)
            
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
            
            # Show entities with possible types
            entities = result['extracted_entities']
            print(f"  Entities ({len(entities)}):")
            for e in entities:
                name = e['entity_name']
                role = e.get('role', 'N/A')
                importance = e.get('importance', 'N/A')
                possible_types = e.get('possible_types', [])
                num_types = len(possible_types)
                
                print(f"    - {name} ({role}/{importance}) - {num_types} types")
                
                stats['total_types_tried'] += num_types
                if num_types > 1:
                    stats['multiple_types_used'] += 1
            
            # Show hybrid retrieval stages
            entity_results = result['retrieval_info']['entity_results']
            
            for entity_result in entity_results:
                entity_name = entity_result['entity_name']
                
                # Stage 1-A
                stage1a_count = entity_result.get('stage1a_value_matches', 0)
                stats['stage1a_total'] += stage1a_count
                
                # Stage 1-B
                stage1b_info = entity_result.get('stage1b_type_info', {})
                type_candidates = stage1b_info.get('type_candidates', 0)
                tried_types = stage1b_info.get('tried_types', [])
                llm_filtered = stage1b_info.get('llm_filtered', 0)
                
                stats['stage1b_type_candidates'] += type_candidates
                stats['stage1b_llm_filtered'] += llm_filtered
                
                # Stage 2
                final_count = entity_result.get('stage2_final', 0)
                stats['stage2_final'] += final_count
                
                print(f"    '{entity_name}':")
                print(f"      1-A (Value): {stage1a_count} matches")
                print(f"      1-B (Type):  {type_candidates} candidates ({len(tried_types)} types tried)")
                if tried_types:
                    types_str = ', '.join(tried_types)  # tried_types is already formatted as "Type/Subtype"
                    print(f"                   Types: {types_str}")
                print(f"      1-B (LLM):   {llm_filtered} filtered")
                print(f"      2 (Final):   {final_count} passages")
                
                # Track interesting multi-type examples
                if len(tried_types) > 1 and type_candidates > 0:
                    type_coverage_examples.append({
                        'entity': entity_name,
                        'tried_types': tried_types,
                        'candidates': type_candidates,
                        'query': query[:50]
                    })
            
            # Check retrieval
            num_passages = len(result['retrieved_passages'])
            
            if num_passages == 0:
                stats['no_match'] += 1
                unmatched_queries.append({
                    'index': i + 1,
                    'type': q_type,
                    'question': query,
                    'reason': 'no_passages',
                    'entities': [e['entity_name'] for e in entities]
                })
                print(f"  [X] No passages retrieved")
                if q_type in type_stats:
                    type_stats[q_type]['no_match'] += 1
            else:
                stats['retrieved'] += 1
                print(f"  [OK] {num_passages} passages:")
                for p in result['retrieved_passages'][:3]:
                    title = p['title']
                    t = p['metadata'].get('type', 'N/A')
                    st = p['metadata'].get('subtype', 'N/A')
                    # Handle potential unicode issues
                    try:
                        print(f"       - {title} ({t}/{st})")
                    except UnicodeEncodeError:
                        print(f"       - [Unicode Title] ({t}/{st})")
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
        
        print(f"\nMultiple Types Feature:")
        print(f"  Entities with multiple types: {stats['multiple_types_used']}")
        print(f"  Total types tried: {stats['total_types_tried']}")
        if stats['multiple_types_used'] > 0:
            avg_types = stats['total_types_tried'] / stats['extraction_success']
            print(f"  Average types per entity: {avg_types:.2f}")
        
        print(f"\nHybrid Retrieval Stages:")
        print(f"  Stage 1-A (Value):  {stats['stage1a_total']} total matches")
        print(f"  Stage 1-B (Type):   {stats['stage1b_type_candidates']} candidates (multiple types)")
        print(f"  Stage 1-B (LLM):    {stats['stage1b_llm_filtered']} filtered")
        print(f"  Stage 2 (Final):    {stats['stage2_final']} passages")
        
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
        
        # Show multi-type examples
        if type_coverage_examples:
            print("\n" + "="*80)
            print(f"MULTIPLE TYPES EXAMPLES (Top 5)")
            print("="*80)
            for i, ex in enumerate(type_coverage_examples[:5], 1):
                print(f"\n{i}. Entity: '{ex['entity']}'")
                print(f"   Query: {ex['query']}...")
                print(f"   Tried types: {', '.join(ex['tried_types'])}")  # Already formatted
                print(f"   Candidates found: {ex['candidates']}")
        
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
        output_file = f'test_hybrid_{num_tests}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_type': 'hybrid_retrieval_multiple_types',
                'num_tests': num_tests,
                'stats': stats,
                'type_stats': type_stats,
                'type_coverage_examples': type_coverage_examples,
                'unmatched_queries': unmatched_queries
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print("="*80)
        
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
