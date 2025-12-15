import json
import sqlite3
from embedding_text_generator import EmbeddingTextGenerator

def test_extraction():
    conn = sqlite3.connect('HotpotQA/metadata_v3.db')
    cursor = conn.cursor()
    cursor.execute('SELECT metadata_json FROM metadata WHERE title = ?', ('Julian McMahon',))
    row = cursor.fetchone()
    
    if not row:
        print("Julian McMahon not found in DB")
        return
        
    metadata = json.loads(row[0])
    
    generator = EmbeddingTextGenerator(language="en")
    results = generator.extract_embedding_texts("Julian McMahon", metadata)
    
    print(f"Extracted {len(results)} paths for Julian McMahon:")
    found_title = False
    for r in results:
        path_str = '.'.join(r['key_path'])
        print(f"  {path_str}: {r['value']}")
        
        if 'parent.title' in path_str or ('parent' in path_str and 'title' in path_str):
            if r['value'] == "former Prime Minister of Australia":
                found_title = True
                
    if found_title:
        print("\nSUCCESS: Found 'former Prime Minister of Australia' in extracted paths!")
    else:
        print("\nFAILURE: Did NOT find 'former Prime Minister of Australia'.")

if __name__ == "__main__":
    test_extraction()
