"""
Sample 200 QA items from MuSiQue (`musique.json`) and keep aligned corpus entries
- Prefer 200 QA that map to 200 distinct corpus titles. If not enough distinct titles, fall back to allowing duplicates.
- Writes:
  - Result/Samples/MuSiQue_qa_sample_200.json
  - Result/Samples/MuSiQue_corpus_aligned_200.json
  - Result/Samples/MuSiQue_sampling_summary.json
"""
from pathlib import Path
import json
import random

ROOT = Path(__file__).parent
OUT_DIR = ROOT.parent / 'Result' / 'Samples'
OUT_DIR.mkdir(parents=True, exist_ok=True)

QA_PATH = ROOT / 'musique.json'
CORPUS_PATH = ROOT / 'musique_corpus.json'

SAMPLE_SIZE = 200
SEED = 42


def load_json(path):
    with path.open('r', encoding='utf-8', errors='replace') as f:
        return json.load(f)


def extract_primary_title(paragraphs):
    # paragraphs might be a list of strings, or list of dicts with 'title' key, or list of {'title':..., 'text':...}
    if not paragraphs:
        return None
    if isinstance(paragraphs, list):
        first = paragraphs[0]
        if isinstance(first, dict):
            # try 'title' key
            return first.get('title') or first.get('doc_id') or None
        elif isinstance(first, str):
            return first
    # otherwise unknown
    return None


def main():
    qa_list = load_json(QA_PATH)
    corpus = load_json(CORPUS_PATH)
    print('Loaded QA:', len(qa_list), 'corpus:', len(corpus))

    # build index for corpus by title (lowercased normalized)
    corpus_by_title = {}
    for c in corpus:
        title = c.get('title')
        if title is None:
            continue
        corpus_by_title.setdefault(title, []).append(c)

    print('Corpus distinct titles:', len(corpus_by_title))

    rnd = random.Random(SEED)
    indices = list(range(len(qa_list)))
    rnd.shuffle(indices)

    chosen_qas = []
    chosen_titles = set()

    # first pass: try to collect QA that map to distinct corpus titles
    for idx in indices:
        q = qa_list[idx]
        paras = q.get('paragraphs')
        prim = extract_primary_title(paras)
        if prim is None:
            continue
        if prim in corpus_by_title and prim not in chosen_titles:
            chosen_qas.append(q)
            chosen_titles.add(prim)
            if len(chosen_qas) >= SAMPLE_SIZE:
                break

    # if not enough distinct titles, fill remaining QA allowing duplicates
    if len(chosen_qas) < SAMPLE_SIZE:
        for idx in indices:
            if len(chosen_qas) >= SAMPLE_SIZE:
                break
            q = qa_list[idx]
            if q in chosen_qas:
                continue
            # ensure we include QA even if primary title absent
            chosen_qas.append(q)
            paras = q.get('paragraphs')
            prim = extract_primary_title(paras)
            if prim:
                chosen_titles.add(prim)

    k = len(chosen_qas)
    print('Selected QA:', k, 'distinct titles:', len(chosen_titles))

    # build aligned corpus list in the same order as chosen_titles list
    # We'll preserve order by iterating through chosen_qas and taking their primary title's first matching corpus doc
    aligned_corpus = []
    used_titles = set()
    for q in chosen_qas:
        paras = q.get('paragraphs')
        prim = extract_primary_title(paras)
        chosen_doc = None
        if prim and prim in corpus_by_title:
            # pick first corpus entry for that title that's not used yet
            for c in corpus_by_title[prim]:
                # we can use the object itself
                if c.get('title') not in used_titles:
                    chosen_doc = c
                    used_titles.add(c.get('title'))
                    break
            if chosen_doc is None:
                # all docs with that title used, just take first
                chosen_doc = corpus_by_title[prim][0]
        else:
            # fallback: pick a random corpus doc
            chosen_doc = rnd.choice(corpus)
        aligned_corpus.append(chosen_doc)

    # If aligned_corpus has fewer than k (shouldn't), pad with random docs
    while len(aligned_corpus) < k:
        aligned_corpus.append(rnd.choice(corpus))

    # write outputs
    qa_out = OUT_DIR / f'MuSiQue_qa_sample_{k}.json'
    corpus_out = OUT_DIR / f'MuSiQue_corpus_aligned_{k}.json'
    summary_out = OUT_DIR / 'MuSiQue_sampling_summary.json'

    with qa_out.open('w', encoding='utf-8') as f:
        json.dump(chosen_qas, f, ensure_ascii=False, indent=2)
    with corpus_out.open('w', encoding='utf-8') as f:
        json.dump(aligned_corpus, f, ensure_ascii=False, indent=2)

    summary = {
        'qa_source': str(QA_PATH),
        'corpus_source': str(CORPUS_PATH),
        'qa_out': str(qa_out),
        'corpus_out': str(corpus_out),
        'requested_sample_size': SAMPLE_SIZE,
        'actual_qas': k,
        'distinct_corpus_titles_in_selection': len(set(c.get('title') for c in aligned_corpus if c.get('title')))
    }
    with summary_out.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('Wrote QA ->', qa_out)
    print('Wrote corpus ->', corpus_out)
    print('Wrote summary ->', summary_out)

if __name__ == '__main__':
    main()
