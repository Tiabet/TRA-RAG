"""
Check if "Education in Argentina" exists in DB
"""
from metadata_db import MetadataDB

db = MetadataDB('metadata_v2.db')

# Search for Argentina-related education passages
print("="*80)
print("Searching for 'Education in Argentina'...")
print("="*80)

# Search by title
result = db.get_by_title("Education in Argentina")
if result:
    print(f"\n✓ Found by title!")
    print(f"  Type: {result['metadata'].get('type')}/{result['metadata'].get('subtype')}")
else:
    print("\n✗ NOT found by exact title")
    
    # Try FTS search
    print("\nTrying FTS search for 'Argentina education'...")
    results = db.search_by_entity_fts("Argentina education", None, None)
    print(f"Results: {len(results)}")
    
    if len(results) > 0:
        print("\nTop 5 results:")
        for i, r in enumerate(results[:5], 1):
            print(f"\n{i}. {r['title']}")
            print(f"   Type: {r['metadata'].get('type')}/{r['metadata'].get('subtype')}")

# Also check what "education" search returns
print("\n" + "="*80)
print("All 23 'education' results:")
print("="*80)
results = db.search_by_entity_fts("education", None, None)
for i, r in enumerate(results, 1):
    title = r['title']
    if 'Argentina' in title or 'Taquini' in title or 'free education' in title.lower():
        print(f"{i}. ✓✓ {title} ← RELEVANT!")
    else:
        print(f"{i}. {title}")

db.close()
