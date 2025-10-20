"""
Test multiple types extraction and retrieval
- Test if new possible_types format works
- Check if "education" now matches Concept/SocialSystem
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from hybrid_retrieval import retrieve_for_query, initialize_llm_client
from metadata_db import MetadataDB


async def main():
    # Initialize
    client = initialize_llm_client()
    db = MetadataDB("metadata_v2.db")
    
    print("="*80)
    print("TEST: Multiple Types Extraction - Argentina Education Query")
    print("="*80)
    
    query = "What role do state institutions play in Argentina's education system?"
    
    print(f"\nQuery: {query}")
    print("-"*80)
    
    # Retrieve
    result = await retrieve_for_query(client, db, query, use_fts=True)
    
    # Display extraction result
    print("\n1️⃣ ENTITY EXTRACTION:")
    print("-"*80)
    
    if result['extraction_result']['success']:
        entities = result['extracted_entities']
        print(f"Extracted {len(entities)} entities:\n")
        
        for i, entity in enumerate(entities, 1):
            print(f"{i}. {entity.get('entity_name')} ({entity.get('role')}, {entity.get('importance')})")
            
            # Display possible types
            possible_types = entity.get('possible_types', [])
            if possible_types:
                print(f"   Possible types ({len(possible_types)}):")
                for j, type_info in enumerate(possible_types, 1):
                    print(f"     {j}) {type_info.get('type')}/{type_info.get('subtype')}")
            else:
                # Old format fallback
                print(f"   Type: {entity.get('type')}/{entity.get('subtype')}")
            print()
    else:
        print(f"❌ Extraction failed: {result['extraction_result'].get('error')}")
    
    # Display retrieval results
    print("\n2️⃣ RETRIEVAL RESULTS:")
    print("-"*80)
    
    retrieval_info = result.get('retrieval_info', {})
    entity_results = retrieval_info.get('entity_results', [])
    
    for i, entity_info in enumerate(entity_results, 1):
        print(f"\n[Entity {i}] {entity_info.get('entity_name')}")
        print(f"  Role: {entity_info.get('entity_role')} ({entity_info.get('entity_importance')})")
        
        # Stage 1-A results
        stage1a_count = entity_info.get('stage1a_value_matches', 0)
        print(f"  Stage 1-A (Value matching): {stage1a_count} passages")
        
        # Stage 1-B results
        stage1b_info = entity_info.get('stage1b_type_info', {})
        type_candidates = stage1b_info.get('type_candidates', 0)
        tried_types = stage1b_info.get('tried_types', [])
        
        print(f"  Stage 1-B (Type filtering):")
        print(f"    Tried types ({len(tried_types)}): {', '.join([f'{t[0]}/{t[1]}' for t in tried_types])}")
        print(f"    Type candidates: {type_candidates}")
        print(f"    LLM filtered: {stage1b_info.get('llm_filtered', 0)}")
        
        # Final results
        final_count = entity_info.get('stage2_final', 0)
        print(f"  Stage 2 (Final): {final_count} passages")
    
    # Display final passages
    print("\n3️⃣ FINAL PASSAGES:")
    print("-"*80)
    
    passages = result.get('retrieved_passages', [])
    print(f"\nTotal: {len(passages)} passages\n")
    
    for i, passage in enumerate(passages[:10], 1):  # Show top 10
        title = passage.get('title', 'N/A')
        type_val = passage.get('metadata', {}).get('type', 'N/A')
        subtype = passage.get('metadata', {}).get('subtype', 'N/A')
        
        print(f"{i}. {title} ({type_val}/{subtype})")
    
    if len(passages) > 10:
        print(f"... and {len(passages) - 10} more passages")
    
    # Check if "Education in Argentina" is present
    print("\n4️⃣ ANSWER CHECK:")
    print("-"*80)
    
    target_titles = ["Education in Argentina", "Taquini Plan", "Free education"]
    found_targets = []
    
    for title in target_titles:
        for i, p in enumerate(passages, 1):
            if title.lower() in p.get('title', '').lower():
                found_targets.append((title, i))
                break
    
    if found_targets:
        print("✅ Found target passages:")
        for title, pos in found_targets:
            print(f"  - '{title}' at position {pos}")
    else:
        print("❌ Target passages not found in top results")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(main())
