"""
Test Hybrid Retrieval System (Multiple Types) - 200 samples
============================================================
Parallel processing with concurrent batch execution
"""
import asyncio
import json
import time
from typing import List, Dict
from hybrid_retrieval import initialize_llm_client, retrieve_for_query
from metadata_db import MetadataDB


async def process_single_query(
    client, 
    db: MetadataDB,
    item: Dict,
    index: int,
    total: int
) -> Dict:
    """Process a single query and return results"""
    query = item['question']
    q_type = item.get('type', 'unknown')
    answer = item.get('answer', 'N/A')  # Ground truth answer
    supporting_facts = item.get('supporting_facts', [])  # Ground truth passages
    
    print(f"[{index}/{total}] Processing: {query[:60]}...")
    
    result = {
        'index': index,
        'question': query,
        'answer': answer,
        'supporting_facts': supporting_facts,
        'type': q_type,
        'success': False,
        'extraction_success': False,
        'num_entities': 0,
        'num_types_total': 0,
        'num_passages': 0,
        'retrieved_passages': [],  # NEW: Store actual retrieved passages
        'entity_results': [],
        'stage1a_total': 0,
        'stage1b_candidates': 0,
        'stage1b_filtered': 0,
        'stage2_final': 0,
        'error': None
    }
    
    try:
        # Retrieve with hybrid approach
        retrieval_result = await retrieve_for_query(client, db, query, use_fts=True)
        
        # Check extraction
        if not retrieval_result['extraction_result']['success']:
            result['error'] = 'extraction_failed'
            return result
        
        result['extraction_success'] = True
        
        # Entities info
        entities = retrieval_result['extracted_entities']
        result['num_entities'] = len(entities)
        
        # Count total types
        for e in entities:
            possible_types = e.get('possible_types', [])
            result['num_types_total'] += len(possible_types)
        
        # Retrieval stages
        entity_results = retrieval_result['retrieval_info']['entity_results']
        
        for entity_result in entity_results:
            stage1a = entity_result.get('stage1a_value_matches', 0)
            stage1b_info = entity_result.get('stage1b_type_info', {})
            stage1b_candidates = stage1b_info.get('type_candidates', 0)
            stage1b_filtered = stage1b_info.get('llm_filtered', 0)
            stage2 = entity_result.get('stage2_final', 0)
            
            result['stage1a_total'] += stage1a
            result['stage1b_candidates'] += stage1b_candidates
            result['stage1b_filtered'] += stage1b_filtered
            result['stage2_final'] += stage2
            
            result['entity_results'].append({
                'entity_name': entity_result['entity_name'],
                'role': entity_result.get('entity_role', 'N/A'),
                'importance': entity_result.get('entity_importance', 'N/A'),
                'stage1a': stage1a,
                'stage1b_candidates': stage1b_candidates,
                'stage1b_filtered': stage1b_filtered,
                'tried_types': stage1b_info.get('tried_types', []),
                'stage2_final': stage2
            })
        
        # Final passages - STORE ACTUAL PASSAGES
        retrieved_passages = retrieval_result['retrieved_passages']
        result['num_passages'] = len(retrieved_passages)
        result['success'] = result['num_passages'] > 0
        
        # Store passage titles and metadata
        result['retrieved_passages'] = [
            {
                'title': p['title'],
                'type': p['metadata'].get('type', 'N/A'),
                'subtype': p['metadata'].get('subtype', 'N/A'),
                'doc_id': p.get('doc_id', 'N/A')
            }
            for p in retrieved_passages
        ]
        
        print(f"  → {result['num_entities']} entities, {result['num_passages']} passages")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  → ERROR: {str(e)[:50]}")
    
    return result


async def process_batch(
    client,
    db: MetadataDB,
    batch: List[tuple],  # [(item, index), ...]
    total: int
) -> List[Dict]:
    """Process a batch of queries concurrently"""
    tasks = [
        process_single_query(client, db, item, index, total)
        for item, index in batch
    ]
    return await asyncio.gather(*tasks)


async def main():
    start_time = time.time()
    
    print("="*80)
    print("HYBRID RETRIEVAL TEST (Multiple Types): 200 queries")
    print("="*80)
    print("Configuration:")
    print("  - Parallel processing: Enabled")
    print("  - Batch size: 10 queries")
    print("  - Pipeline: Value matching + Multiple type filtering + LLM filtering")
    print("="*80)
    
    # Initialize
    client = initialize_llm_client()
    db = MetadataDB('metadata_v2.db')
    
    try:
        # Load test queries
        with open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        total_queries = len(test_data)
        print(f"\nLoaded {total_queries} queries")
        
        # Batch processing
        batch_size = 10  # Process 10 queries concurrently
        all_results = []
        
        print(f"\nProcessing in batches of {batch_size}...")
        print("-" * 80)
        
        for i in range(0, total_queries, batch_size):
            batch_end = min(i + batch_size, total_queries)
            batch = [(test_data[j], j + 1) for j in range(i, batch_end)]
            
            print(f"\nBatch {i//batch_size + 1}/{(total_queries + batch_size - 1)//batch_size} (Queries {i+1}-{batch_end})")
            
            batch_results = await process_batch(client, db, batch, total_queries)
            all_results.extend(batch_results)
            
            # Progress indicator
            completed = len(all_results)
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_queries - completed) / rate if rate > 0 else 0
            
            print(f"Progress: {completed}/{total_queries} ({completed/total_queries*100:.1f}%) | "
                  f"Rate: {rate:.2f} q/s | ETA: {eta/60:.1f} min")
        
        # Calculate statistics
        print("\n" + "="*80)
        print("CALCULATING STATISTICS...")
        print("="*80)
        
        stats = {
            'total': len(all_results),
            'extraction_success': sum(1 for r in all_results if r['extraction_success']),
            'extraction_failed': sum(1 for r in all_results if not r['extraction_success']),
            'retrieved': sum(1 for r in all_results if r['success']),
            'no_match': sum(1 for r in all_results if not r['success']),
            'total_entities': sum(r['num_entities'] for r in all_results),
            'total_types': sum(r['num_types_total'] for r in all_results),
            'stage1a_total': sum(r['stage1a_total'] for r in all_results),
            'stage1b_candidates': sum(r['stage1b_candidates'] for r in all_results),
            'stage1b_filtered': sum(r['stage1b_filtered'] for r in all_results),
            'stage2_final': sum(r['stage2_final'] for r in all_results),
            'errors': sum(1 for r in all_results if r['error'])
        }
        
        # Question type statistics
        type_stats = {
            'bridge': {'total': 0, 'retrieved': 0, 'no_match': 0},
            'comparison': {'total': 0, 'retrieved': 0, 'no_match': 0}
        }
        
        for r in all_results:
            q_type = r['type']
            if q_type in type_stats:
                type_stats[q_type]['total'] += 1
                if r['success']:
                    type_stats[q_type]['retrieved'] += 1
                else:
                    type_stats[q_type]['no_match'] += 1
        
        # Calculate recall (how many supporting facts were retrieved)
        recall_stats = {
            'total_supporting_facts': 0,
            'retrieved_supporting_facts': 0,
            'queries_with_full_recall': 0,
            'queries_with_partial_recall': 0,
            'queries_with_no_recall': 0
        }
        
        for r in all_results:
            if not r['success']:
                continue
            
            supporting_titles = [sf[0] for sf in r['supporting_facts']]
            retrieved_titles = [p['title'] for p in r['retrieved_passages']]
            
            recall_stats['total_supporting_facts'] += len(supporting_titles)
            
            matches = sum(1 for st in supporting_titles if st in retrieved_titles)
            recall_stats['retrieved_supporting_facts'] += matches
            
            if matches == len(supporting_titles) and len(supporting_titles) > 0:
                recall_stats['queries_with_full_recall'] += 1
            elif matches > 0:
                recall_stats['queries_with_partial_recall'] += 1
            elif len(supporting_titles) > 0:
                recall_stats['queries_with_no_recall'] += 1
        
        # Multiple types usage
        multi_type_entities = 0
        for r in all_results:
            for entity_result in r['entity_results']:
                if len(entity_result.get('tried_types', [])) > 1:
                    multi_type_entities += 1
        
        # Print summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nExecution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Processing Rate: {stats['total']/elapsed_time:.2f} queries/second")
        
        print(f"\nTotal queries: {stats['total']}")
        
        print(f"\nEntity Extraction:")
        print(f"  Success: {stats['extraction_success']} ({stats['extraction_success']/stats['total']*100:.1f}%)")
        print(f"  Failed:  {stats['extraction_failed']} ({stats['extraction_failed']/stats['total']*100:.1f}%)")
        
        print(f"\nRetrieval Results:")
        print(f"  Retrieved:  {stats['retrieved']} ({stats['retrieved']/stats['total']*100:.1f}%)")
        print(f"  No match:   {stats['no_match']} ({stats['no_match']/stats['total']*100:.1f}%)")
        
        print(f"\nMultiple Types Feature:")
        print(f"  Total entities: {stats['total_entities']}")
        print(f"  Total types tried: {stats['total_types']}")
        if stats['total_entities'] > 0:
            print(f"  Average types per entity: {stats['total_types']/stats['total_entities']:.2f}")
        print(f"  Entities using multiple types: {multi_type_entities}")
        
        print(f"\nHybrid Retrieval Stages:")
        print(f"  Stage 1-A (Value):     {stats['stage1a_total']} matches")
        print(f"  Stage 1-B (Type):      {stats['stage1b_candidates']} candidates (multiple types)")
        print(f"  Stage 1-B (LLM):       {stats['stage1b_filtered']} filtered")
        if stats['stage1b_candidates'] > 0:
            filter_rate = (stats['stage1b_candidates'] - stats['stage1b_filtered']) / stats['stage1b_candidates'] * 100
            print(f"  LLM filter rate:       {filter_rate:.1f}% removed")
        print(f"  Stage 2 (Final):       {stats['stage2_final']} passages")
        
        print(f"\nErrors: {stats['errors']}")
        
        # Recall analysis
        print("\n" + "="*80)
        print("RECALL ANALYSIS (Supporting Facts)")
        print("="*80)
        
        if recall_stats['total_supporting_facts'] > 0:
            recall_rate = recall_stats['retrieved_supporting_facts'] / recall_stats['total_supporting_facts'] * 100
            print(f"\nOverall Recall:")
            print(f"  Total supporting facts: {recall_stats['total_supporting_facts']}")
            print(f"  Retrieved: {recall_stats['retrieved_supporting_facts']} ({recall_rate:.1f}%)")
            
            print(f"\nQuery-level Recall:")
            successful_queries = stats['retrieved']
            if successful_queries > 0:
                print(f"  Full recall (all supporting facts): {recall_stats['queries_with_full_recall']} ({recall_stats['queries_with_full_recall']/successful_queries*100:.1f}%)")
                print(f"  Partial recall (some facts): {recall_stats['queries_with_partial_recall']} ({recall_stats['queries_with_partial_recall']/successful_queries*100:.1f}%)")
                print(f"  No recall: {recall_stats['queries_with_no_recall']} ({recall_stats['queries_with_no_recall']/successful_queries*100:.1f}%)")
        
        # Question type analysis
        print("\n" + "="*80)
        print("QUESTION TYPE ANALYSIS")
        print("="*80)
        
        for q_type in ['bridge', 'comparison']:
            if type_stats[q_type]['total'] == 0:
                continue
            ts = type_stats[q_type]
            print(f"\n{q_type.upper()}:")
            print(f"  Total:     {ts['total']}")
            print(f"  Retrieved: {ts['retrieved']} ({ts['retrieved']/ts['total']*100:.1f}%)")
            print(f"  No match:  {ts['no_match']} ({ts['no_match']/ts['total']*100:.1f}%)")
        
        # Find interesting cases
        print("\n" + "="*80)
        print("INTERESTING CASES")
        print("="*80)
        
        # High type candidates
        high_candidates = sorted(
            [r for r in all_results if r['stage1b_candidates'] > 50],
            key=lambda x: x['stage1b_candidates'],
            reverse=True
        )[:5]
        
        if high_candidates:
            print("\nTop 5: Highest Type Candidates")
            for i, r in enumerate(high_candidates, 1):
                print(f"{i}. [{r['index']}] {r['question'][:60]}...")
                print(f"   Type candidates: {r['stage1b_candidates']} → LLM filtered: {r['stage1b_filtered']}")
        
        # High LLM filtering
        high_filtering = sorted(
            [r for r in all_results if r['stage1b_candidates'] > 0],
            key=lambda x: (r['stage1b_candidates'] - r['stage1b_filtered']) / r['stage1b_candidates'] if r['stage1b_candidates'] > 0 else 0,
            reverse=True
        )[:5]
        
        if high_filtering:
            print("\nTop 5: Highest LLM Filtering Rate")
            for i, r in enumerate(high_filtering, 1):
                if r['stage1b_candidates'] > 0:
                    filter_rate = (r['stage1b_candidates'] - r['stage1b_filtered']) / r['stage1b_candidates'] * 100
                    print(f"{i}. [{r['index']}] {r['question'][:60]}...")
                    print(f"   {r['stage1b_candidates']} → {r['stage1b_filtered']} ({filter_rate:.1f}% filtered)")
        
        # Save detailed results
        output_file = 'test_hybrid_200_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_type': 'hybrid_retrieval_multiple_types_200',
                'execution_time': elapsed_time,
                'stats': stats,
                'type_stats': type_stats,
                'recall_stats': recall_stats,
                'all_results': all_results
            }, f, indent=2, ensure_ascii=False)
        
        # Also save a summary file with just key info
        summary_file = 'test_hybrid_200_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HYBRID RETRIEVAL TEST - 200 QUERIES SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Execution Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)\n")
            f.write(f"Processing Rate: {stats['total']/elapsed_time:.2f} q/s\n\n")
            
            f.write(f"Retrieval Success: {stats['retrieved']}/{stats['total']} ({stats['retrieved']/stats['total']*100:.1f}%)\n")
            f.write(f"Recall Rate: {recall_stats['retrieved_supporting_facts']}/{recall_stats['total_supporting_facts']} ({recall_stats['retrieved_supporting_facts']/recall_stats['total_supporting_facts']*100:.1f}%)\n\n")
            
            f.write("Multiple Types Feature:\n")
            f.write(f"  Average types per entity: {stats['total_types']/stats['total_entities']:.2f}\n")
            f.write(f"  Entities using multiple types: {multi_type_entities}\n\n")
            
            f.write("="*80 + "\n")
            f.write("SAMPLE RESULTS (First 10)\n")
            f.write("="*80 + "\n\n")
            
            for i, r in enumerate(all_results[:10], 1):
                f.write(f"{i}. [{r['index']}] {r['question']}\n")
                f.write(f"   Type: {r['type']}\n")
                f.write(f"   Success: {r['success']} | Passages: {r['num_passages']}\n")
                
                if r['supporting_facts']:
                    supporting_titles = [sf[0] for sf in r['supporting_facts']]
                    retrieved_titles = [p['title'] for p in r['retrieved_passages']]
                    matches = [st for st in supporting_titles if st in retrieved_titles]
                    
                    f.write(f"   Supporting facts: {supporting_titles}\n")
                    f.write(f"   Retrieved (top 5): {retrieved_titles[:5]}\n")
                    f.write(f"   Matches: {matches} ({len(matches)}/{len(supporting_titles)})\n")
                
                f.write("\n")
        
        print(f"\n{'='*80}")
        print(f"Results saved to:")
        print(f"  - Detailed: {output_file}")
        print(f"  - Summary: {summary_file}")
        print("="*80)
        
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
