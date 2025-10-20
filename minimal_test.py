"""
Minimal test: Just check type candidates count
No LLM filtering to save time
"""
from metadata_db import MetadataDB

def test_type_search():
    db = MetadataDB("metadata_v2.db")
    
    print("="*80)
    print("MINIMAL TEST: Type Matching (DB only, no LLM)")
    print("="*80)
    
    # Test "education system" with multiple types
    entity_name = "education system"
    possible_types = [
        {"type": "Concept", "subtype": "EducationalSystem"},
        {"type": "Concept", "subtype": "SocialSystem"},
        {"type": "Concept", "subtype": "AcademicField"}
    ]
    
    print(f"\nEntity: {entity_name}")
    print(f"Trying {len(possible_types)} type combinations:")
    
    all_candidates = []
    seen_titles = set()
    
    for i, type_info in enumerate(possible_types, 1):
        t = type_info['type']
        st = type_info['subtype']
        
        candidates = db.search_by_type(t, st)
        
        # Deduplicate by title
        new_count = 0
        for c in candidates:
            title = c['title']
            if title not in seen_titles:
                seen_titles.add(title)
                all_candidates.append(c)
                new_count += 1
        
        print(f"  {i}) {t}/{st}: {len(candidates)} found, {new_count} new")
    
    print(f"\nTotal unique candidates: {len(all_candidates)}")
    
    # Check if target answers are in candidates
    print("\nTarget answers in candidates:")
    targets = ["Education in Argentina", "Taquini Plan", "Free education"]
    for target in targets:
        found = any(target.lower() in c['title'].lower() for c in all_candidates)
        status = "✅" if found else "❌"
        print(f"  {status} {target}")
    
    # Show sample titles
    print(f"\nSample candidates (first 10):")
    for i, c in enumerate(all_candidates[:10], 1):
        title = c['title']
        t = c['metadata'].get('type', 'N/A')
        st = c['metadata'].get('subtype', 'N/A')
        print(f"  {i}. {title} ({t}/{st})")

if __name__ == "__main__":
    test_type_search()
