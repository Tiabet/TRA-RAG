import json
from pathlib import Path

ROOT = Path(__file__).parent
qa_path = ROOT / 'musique.json'
corpus_path = ROOT / 'musique_corpus.json'

def load(path):
    print('Loading', path)
    with path.open('r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
    print('Type:', type(data), 'len:', len(data) if isinstance(data, list) else 'N/A')
    return data

qa = load(qa_path)
corpus = load(corpus_path)

print('\n--- QA sample keys (up to 5) ---')
for i, it in enumerate(qa[:5]):
    print(i, list(it.keys()))

print('\n--- Corpus sample keys (up to 5) ---')
for i, it in enumerate(corpus[:5]):
    print(i, list(it.keys()))

# try to detect common id-like keys
qa_keys = set().union(*(set(x.keys()) for x in qa))
corpus_keys = set().union(*(set(x.keys()) for x in corpus))
print('\nQA keys union sample count:', len(qa_keys))
print('Corpus keys union sample count:', len(corpus_keys))
print('\nSome QA keys\n', sorted(list(qa_keys))[:50])
print('\nSome Corpus keys\n', sorted(list(corpus_keys))[:50])

# Candidate matches: for small subset, look for equality of values
print('\nSearching for matching field values between QA and corpus (first 200 items)')
qa_sample = qa[:200]
corpus_sample = corpus[:200]

matches = []
for q in qa_sample:
    for c in corpus_sample:
        # compare simple keys
        for kq, vq in q.items():
            for kc, vc in c.items():
                if vq == vc and not isinstance(vq, (list, dict)):
                    matches.append((kq, kc, vq))
                    if len(matches) > 20:
                        break
            if len(matches) > 20:
                break
        if len(matches) > 20:
            break
    if len(matches) > 20:
        break

print('Found matches (up to 20):')
for m in matches[:20]:
    print(m)

# print some example values from fields that look like ids
def show_field_values(name, arr, field, n=5):
    vals = [x.get(field) for x in arr if field in x]
    print(f"{name} field '{field}' sample (count {len(vals)}):", vals[:n])

candidates = ['id','qid','qid_str','question_id','corpus_id','context_id','source_id']
for cand in candidates:
    if any(cand in x for x in qa):
        show_field_values('QA', qa, cand)
    if any(cand in x for x in corpus):
        show_field_values('Corpus', corpus, cand)

print('\nDone')
