import json
from pathlib import Path

def analyze(path):
    p = Path(path)
    if not p.exists():
        print(f"{path}: NOT FOUND")
        return
    s = p.read_text(encoding='utf-8')
    try:
        obj = json.loads(s)
    except Exception as e:
        print(f"{path}: JSON LOAD ERROR: {e}")
        return
    if isinstance(obj, list):
        n = len(obj)
        total_chars = 0
        total_words = 0
        titles = []
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                text = item.get('text') or item.get('content') or ''
                title = item.get('title') or item.get('id') or ''
                titles.append(title)
            else:
                text = str(item)
            total_chars += len(text)
            total_words += len(text.split())
        avg_chars = total_chars / n if n else 0
        avg_words = total_words / n if n else 0
        print(f"{path}: list with {n} items")
        print(f"  total chars: {total_chars:,}, total words: {total_words:,}")
        print(f"  avg chars/item: {avg_chars:.1f}, avg words/item: {avg_words:.1f}")
        print(f"  sample titles (first 10): {titles[:10]}")
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{path}: dict with {len(keys)} keys; sample keys: {keys[:20]}")
    else:
        print(f"{path}: JSON type: {type(obj).__name__}")

if __name__ == '__main__':
    analyze('c:/Development/ChunkRAG/MuSiQue/musique_corpus.json')
    analyze('c:/Development/ChunkRAG/MuSiQue/musique.json')
