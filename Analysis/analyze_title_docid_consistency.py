import json
from pathlib import Path


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_results(path: str):
    data = load_json(path)
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    return data


def build_docid_to_title(gold_path: str):
    gold = load_json(gold_path)
    mapping = {}
    for item in gold:
        sid = item.get('_id')
        if not sid:
            continue
        for ctx_idx, (title, _sentences) in enumerate(item.get('context', [])):
            mapping[f"{sid}::ctx{ctx_idx}"] = title
    return mapping


def iter_retrieved_passages(result_item):
    # top-level
    for p in (result_item.get('retrieved_passages') or []):
        if isinstance(p, dict):
            yield p

    decomp = result_item.get('decomposition')
    subqs = []
    if isinstance(decomp, list):
        subqs = decomp
    elif isinstance(decomp, dict):
        subqs = decomp.get('subquestions') or []

    for sq in subqs:
        for p in (sq.get('retrieved_passages') or []):
            if isinstance(p, dict):
                yield p


def analyze(result_path: str, gold_path: str):
    results = load_results(result_path)
    docid_to_title = build_docid_to_title(gold_path)

    checked = 0
    match = 0
    missing_docid = 0
    missing_gold_lookup = 0

    for r in results:
        for p in iter_retrieved_passages(r):
            doc_id = p.get('doc_id')
            title = p.get('title')
            if not doc_id:
                missing_docid += 1
                continue
            doc_id = str(doc_id)
            gold_title = docid_to_title.get(doc_id)
            if gold_title is None:
                missing_gold_lookup += 1
                continue
            checked += 1
            if str(title) == str(gold_title):
                match += 1

    print(f"\n{result_path}")
    print(f"  checked_passages: {checked}")
    print(f"  title_matches_docid_title: {match} ({(match/checked if checked else 0):.4f})")
    print(f"  missing_doc_id_in_passage: {missing_docid}")
    print(f"  doc_id_not_in_gold_map: {missing_gold_lookup}")


def main():
    gold_path = 'HotpotQA/hotpotqa_sample_200.json'
    analyze('Results/test_hotpot_v11_200_results.json', gold_path)
    analyze('Results/test_hotpot_v11_200_results_v2.json', gold_path)


if __name__ == '__main__':
    main()
