"""
Quick test: Check type matching with multiple types
Focus on "education system" entity only
"""
import asyncio
import json
from hybrid_retrieval import initialize_llm_client, stage1b_type_filtering
from metadata_db import MetadataDB

async def main():
    client = initialize_llm_client()
    db = MetadataDB("metadata_v2.db")
    
    print("="*80)
    print("QUICK TEST: Type Matching with Multiple Types")
    print("="*80)
    
    query = "What role do state institutions play in Argentina's education system?"
    entity_name = "education system"
    
    # Multiple possible types (from extraction)
    possible_types = [
        {"type": "Concept", "subtype": "EducationalSystem"},
        {"type": "Concept", "subtype": "SocialSystem"},
        {"type": "Concept", "subtype": "AcademicField"}
    ]
    
    print(f"\nQuery: {query}")
    print(f"Entity: {entity_name}")
    print(f"Possible types: {len(possible_types)}")
    for i, t in enumerate(possible_types, 1):
        print(f"  {i}) {t['type']}/{t['subtype']}")
    
    print("\n" + "-"*80)
    print("Running Stage 1-B (Type filtering + LLM)...")
    print("-"*80)
    
    try:
        type_matches, type_info = await stage1b_type_filtering(
            client, db, query, entity_name, possible_types
        )
        
        print("\nResults:")
        print(f"  Type candidates found: {type_info.get('type_candidates', 0)}")
        print(f"  Tried types: {type_info.get('tried_types', [])}")
        print(f"  LLM filtered: {type_info.get('llm_filtered', 0)}")
        print(f"  Final matches: {len(type_matches)}")
        
        if type_matches:
            print("\n✅ Top 5 matches:")
            for i, match in enumerate(type_matches[:5], 1):
                title = match.get('title', 'N/A')
                t = match['metadata'].get('type', 'N/A')
                st = match['metadata'].get('subtype', 'N/A')
                print(f"  {i}. {title} ({t}/{st})")
        else:
            print("\n❌ No matches found")
        
        # Check if "Education in Argentina" is in results
        print("\n" + "-"*80)
        print("Answer check:")
        print("-"*80)
        
        target_titles = ["Education in Argentina", "Taquini Plan", "Free education"]
        for target in target_titles:
            found = any(target.lower() in m.get('title', '').lower() for m in type_matches)
            status = "✅" if found else "❌"
            print(f"  {status} {target}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
