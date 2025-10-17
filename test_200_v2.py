"""
Comprehensive Test with V2 Metadata: 200 queries
================================================
Tests with improved metadata (no information loss)
"""
import asyncio
import json
from db_entity_retrieval import initialize_llm_client, retrieve_for_query
from metadata_db import MetadataDB


async def main():
    client = initialize_llm_client()
    db = MetadataDB('metadata_v2.db')  # Use V2 database
    
    try:
        # Load test queries
        with open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print("="*80)
        print(f"COMPREHENSIVE TEST with V2 METADATA: {len(test_data)} queries")
        print("="*80)
        
        # Overall statistics
        overall_stats = {
            'total': 0,
            'extraction_success': 0,
            'extraction_failed': 0,
            'retrieved': 0,
            'no_match': 0
        }
        
        # Question type statistics
        type_stats = {
            'bridge': {
                'total': 0,
                'retrieved': 0,
                'no_match': 0,
                'avg_entities': [],
                'avg_passages': []
            },
            'comparison': {
                'total': 0,
                'retrieved': 0,
                'no_match': 0,
                'avg_entities': [],
                'avg_passages': []
            }
        }
        
        # Track unmatched queries
        unmatched_queries = []
        
        print("\nProcessing queries...")
        print("-" * 80)
        
        for i, item in enumerate(test_data):
            query = item['question']
            q_type = item.get('type', 'unknown')
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(test_data)} ({(i+1)/len(test_data)*100:.1f}%)")
            
            overall_stats['total'] += 1
            
            # Track by type
            if q_type in type_stats:
                type_stats[q_type]['total'] += 1
            
            # Retrieve
            result = await retrieve_for_query(client, db, query)
            
            # Check extraction
            if not result['extraction_result']['success']:
                overall_stats['extraction_failed'] += 1
                unmatched_queries.append({
                    'index': i + 1,
                    'type': q_type,
                    'question': query,
                    'reason': 'extraction_failed',
                    'error': result['extraction_result'].get('error', 'Unknown')
                })
                if q_type in type_stats:
                    type_stats[q_type]['no_match'] += 1
                continue
            
            overall_stats['extraction_success'] += 1
            
            # Check retrieval
            num_entities = len(result['extracted_entities'])
            num_passages = len(result['retrieved_passages'])
            
            if num_passages == 0:
                overall_stats['no_match'] += 1
                unmatched_queries.append({
                    'index': i + 1,
                    'type': q_type,
                    'question': query,
                    'reason': 'no_passages',
                    'entities': [e['entity_name'] for e in result['extracted_entities']]
                })
                if q_type in type_stats:
                    type_stats[q_type]['no_match'] += 1
                    type_stats[q_type]['avg_entities'].append(num_entities)
                    type_stats[q_type]['avg_passages'].append(0)
            else:
                overall_stats['retrieved'] += 1
                if q_type in type_stats:
                    type_stats[q_type]['retrieved'] += 1
                    type_stats[q_type]['avg_entities'].append(num_entities)
                    type_stats[q_type]['avg_passages'].append(num_passages)
        
        # Print results
        print("\n" + "="*80)
        print("OVERALL STATISTICS (V2 METADATA)")
        print("="*80)
        print(f"Total queries: {overall_stats['total']}")
        print(f"\nEntity Extraction:")
        print(f"  Success: {overall_stats['extraction_success']} ({overall_stats['extraction_success']/overall_stats['total']*100:.1f}%)")
        print(f"  Failed:  {overall_stats['extraction_failed']} ({overall_stats['extraction_failed']/overall_stats['total']*100:.1f}%)")
        
        print(f"\nRetrieval Results:")
        print(f"  Retrieved:  {overall_stats['retrieved']} ({overall_stats['retrieved']/overall_stats['total']*100:.1f}%)")
        print(f"  No match:   {overall_stats['no_match']} ({overall_stats['no_match']/overall_stats['total']*100:.1f}%)")
        
        # Question type analysis
        print("\n" + "="*80)
        print("QUESTION TYPE ANALYSIS")
        print("="*80)
        
        for q_type in ['bridge', 'comparison']:
            stats = type_stats[q_type]
            if stats['total'] == 0:
                continue
            
            print(f"\n{q_type.upper()} Questions:")
            print(f"  Total: {stats['total']}")
            print(f"  Retrieved: {stats['retrieved']} ({stats['retrieved']/stats['total']*100:.1f}%)")
            print(f"  No match:  {stats['no_match']} ({stats['no_match']/stats['total']*100:.1f}%)")
            
            if stats['avg_entities']:
                avg_entities = sum(stats['avg_entities']) / len(stats['avg_entities'])
                print(f"  Avg entities per query: {avg_entities:.2f}")
            
            if stats['avg_passages']:
                avg_passages = sum(stats['avg_passages']) / len(stats['avg_passages'])
                print(f"  Avg passages retrieved: {avg_passages:.2f}")
        
        # Unmatched queries
        print("\n" + "="*80)
        print(f"UNMATCHED QUERIES: {len(unmatched_queries)}")
        print("="*80)
        
        # Group by reason
        by_reason = {}
        for uq in unmatched_queries:
            reason = uq['reason']
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(uq)
        
        print(f"\nBy Reason:")
        for reason, queries in by_reason.items():
            print(f"  {reason}: {len(queries)}")
        
        # Show first 10 unmatched
        print(f"\nFirst 10 unmatched queries:")
        for i, uq in enumerate(unmatched_queries[:10]):
            print(f"\n  [{uq['index']}] ({uq['type']}) {uq['question'][:70]}...")
            print(f"      Reason: {uq['reason']}")
            if 'entities' in uq:
                print(f"      Entities: {uq['entities']}")
            if 'error' in uq:
                print(f"      Error: {uq['error']}")
        
        # Save detailed results
        output_file = 'test_results_200_v2.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata_version': 'v2',
                'overall_stats': overall_stats,
                'type_stats': {
                    k: {
                        'total': v['total'],
                        'retrieved': v['retrieved'],
                        'no_match': v['no_match'],
                        'avg_entities': sum(v['avg_entities'])/len(v['avg_entities']) if v['avg_entities'] else 0,
                        'avg_passages': sum(v['avg_passages'])/len(v['avg_passages']) if v['avg_passages'] else 0
                    }
                    for k, v in type_stats.items()
                },
                'unmatched_queries': unmatched_queries
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n" + "="*80)
        print(f"Detailed results saved to: {output_file}")
        print("="*80)
        
        # Summary comparison
        print("\n" + "="*80)
        print("BRIDGE vs COMPARISON")
        print("="*80)
        
        bridge = type_stats['bridge']
        comp = type_stats['comparison']
        
        print(f"\n{'Metric':<30} {'Bridge':<15} {'Comparison':<15}")
        print("-" * 60)
        print(f"{'Total questions':<30} {bridge['total']:<15} {comp['total']:<15}")
        print(f"{'Retrieved (%)':<30} {bridge['retrieved']/bridge['total']*100 if bridge['total'] else 0:<15.1f} {comp['retrieved']/comp['total']*100 if comp['total'] else 0:<15.1f}")
        print(f"{'No match (%)':<30} {bridge['no_match']/bridge['total']*100 if bridge['total'] else 0:<15.1f} {comp['no_match']/comp['total']*100 if comp['total'] else 0:<15.1f}")
        
        if bridge['avg_entities']:
            bridge_avg_ent = sum(bridge['avg_entities'])/len(bridge['avg_entities'])
        else:
            bridge_avg_ent = 0
            
        if comp['avg_entities']:
            comp_avg_ent = sum(comp['avg_entities'])/len(comp['avg_entities'])
        else:
            comp_avg_ent = 0
        
        print(f"{'Avg entities':<30} {bridge_avg_ent:<15.2f} {comp_avg_ent:<15.2f}")
        
        if bridge['avg_passages']:
            bridge_avg_pass = sum(bridge['avg_passages'])/len(bridge['avg_passages'])
        else:
            bridge_avg_pass = 0
            
        if comp['avg_passages']:
            comp_avg_pass = sum(comp['avg_passages'])/len(comp['avg_passages'])
        else:
            comp_avg_pass = 0
        
        print(f"{'Avg passages':<30} {bridge_avg_pass:<15.2f} {comp_avg_pass:<15.2f}")
        
        print("\n" + "="*80)
        print("✨ Test complete with V2 metadata!")
        print("="*80)
        
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
