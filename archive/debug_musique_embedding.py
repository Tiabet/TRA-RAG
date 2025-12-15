import json
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from embedding_text_generator import EmbeddingTextGenerator

def debug_musique_generation():
    # Load a sample from MuSiQue metadata
    with open('MuSiQue/musique_sample_200_metadata.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find an interesting item (e.g., with nested attributes)
    # "Rialto Bridge" from the context looks good (id: 3hop1__208108_547811_80702)
    target_id = "3hop1__208108_547811_80702"
    item = next((x for x in data if x['id'] == target_id), None)
    
    if not item:
        print(f"Item {target_id} not found.")
        return

    print(f"Found item: {item['question']}")
    
    # Extract one context metadata item
    # Let's look for "Rialto Bridge" in context_metadata
    rialto = None
    for ctx in item['context_metadata']:
        if ctx['title'] == "Rialto Bridge":
            rialto = ctx['metadata']
            break
    
    if not rialto:
        print("Rialto Bridge metadata not found.")
        return

    print("\nOriginal Metadata (Rialto Bridge):")
    print(json.dumps(rialto, indent=2, ensure_ascii=False))

    # Generate embedding texts
    generator = EmbeddingTextGenerator(language="en")
    texts = generator.extract_embedding_texts(rialto['title'], rialto)

    print("\nGenerated Embedding Texts:")
    for t in texts:
        print(f"- {t['text']}")

if __name__ == "__main__":
    debug_musique_generation()
