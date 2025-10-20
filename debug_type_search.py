"""
Debug why Type search returns 0 results for "education"
"""
from metadata_db import MetadataDB

db = MetadataDB('metadata_v2.db')

# Check what LLM extracted
entity_type = "Concept"
entity_subtype = "AcademicField"

print("="*80)
print(f"Searching for type: {entity_type}, subtype: {entity_subtype}")
print("="*80)

# Search by type
results = db.search_by_type(entity_type, entity_subtype)
print(f"\nResults: {len(results)}")

if len(results) == 0:
    print("\n❌ No results found!")
    print("\nLet's check what types/subtypes exist in DB:")
    
    # Get all Concept types
    cursor = db.cursor
    cursor.execute("""
        SELECT DISTINCT subtype, COUNT(*) as count
        FROM metadata
        WHERE type = 'Concept'
        GROUP BY subtype
        ORDER BY count DESC
    """)
    
    concept_subtypes = cursor.fetchall()
    print(f"\nConcept subtypes in DB:")
    for row in concept_subtypes:
        print(f"  - {row['subtype']}: {row['count']}")
    
    print("\n" + "="*80)
    print("Now let's search for 'education' by VALUE (Stage 1-A):")
    print("="*80)
    
    value_results = db.search_by_entity_fts("education", None, None)
    print(f"\nValue search results: {len(value_results)}")
    
    if len(value_results) > 0:
        print("\nFirst 5 results:")
        for i, r in enumerate(value_results[:5], 1):
            metadata = r['metadata']
            print(f"\n{i}. {r['title']}")
            print(f"   Type: {metadata.get('type')}/{metadata.get('subtype')}")
            if 'attributes' in metadata and 'description' in metadata['attributes']:
                desc = metadata['attributes']['description']
                print(f"   Description: {desc[:100]}...")

db.close()
