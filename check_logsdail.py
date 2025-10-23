import sqlite3
import json

conn = sqlite3.connect('metadata_v2.db')
cursor = conn.cursor()

# FTS search for Leonard Logsdail
print("=== FTS Search for 'Leonard Logsdail' ===")
cursor.execute("""
    SELECT m.title, m.type, m.subtype, m.metadata_json 
    FROM metadata m
    JOIN metadata_fts fts ON m.id = fts.rowid
    WHERE metadata_fts MATCH ?
""", ('Leonard Logsdail',))
results = cursor.fetchall()
print(f"Found {len(results)} results\n")

for r in results:
    title, typ, subtyp, meta_json = r
    meta = json.loads(meta_json) if meta_json else {}
    print(f"Title: {title}")
    print(f"Type: {typ}/{subtyp}")
    print(f"Attributes: {meta.get('attributes', {})}")
    print(f"Relations: {meta.get('relations', {})}")
    print()

# Also try searching for "Wolf of Wall Street"
print("\n=== FTS Search for 'Wolf of Wall Street' ===")
cursor.execute("""
    SELECT m.title, m.type, m.subtype, m.metadata_json 
    FROM metadata m
    JOIN metadata_fts fts ON m.id = fts.rowid
    WHERE metadata_fts MATCH ?
""", ('Wolf Wall Street',))
results = cursor.fetchall()
print(f"Found {len(results)} results\n")

for r in results:
    title, typ, subtyp, meta_json = r
    meta = json.loads(meta_json) if meta_json else {}
    print(f"Title: {title}")
    print(f"Type: {typ}/{subtyp}")
    print(f"Full Metadata Keys: {list(meta.keys())}")
    desc = meta.get('description', 'N/A')
    print(f"Description: {desc[:500]}")
    full_text = str(meta)
    if 'Leonard' in full_text or 'Logsdail' in full_text:
        print("  *** Contains Leonard/Logsdail! ***")
    print()

conn.close()
