import json
from pathlib import Path


def load_results(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data


def get_retrieved_doc_ids(result_item):
    doc_ids = set()

    def add(v):
        if not v:
            return
        doc_ids.add(str(v))

    for p in (result_item.get("retrieved_passages") or []):
        if isinstance(p, dict):
            if "doc_id" in p:
                add(p.get("doc_id"))
            if isinstance(p.get("doc_ids"), list):
                for d in p.get("doc_ids"):
                    add(d)

    decomp = result_item.get("decomposition")
    subqs = []
    if isinstance(decomp, list):
        subqs = decomp
    elif isinstance(decomp, dict):
        subqs = decomp.get("subquestions") or []

    for sq in subqs:
        for p in (sq.get("retrieved_passages") or []):
            if isinstance(p, dict):
                if "doc_id" in p:
                    add(p.get("doc_id"))
                if isinstance(p.get("doc_ids"), list):
                    for d in p.get("doc_ids"):
                        add(d)
        for rp in (sq.get("retrieved_paths") or []):
            if isinstance(rp, dict) and "doc_id" in rp:
                add(rp.get("doc_id"))

    for rp in (result_item.get("retrieved_paths") or []):
        if isinstance(rp, dict) and "doc_id" in rp:
            add(rp.get("doc_id"))

    return doc_ids


def prefix_match_stats(results):
    ratios = []
    empty = 0

    for r in results:
        qid = r.get("id") or r.get("_id")
        if not qid:
            continue
        doc_ids = get_retrieved_doc_ids(r)
        if not doc_ids:
            empty += 1
            continue
        match = sum(1 for d in doc_ids if d.startswith(str(qid) + "::"))
        ratios.append(match / len(doc_ids))

    ratios_sorted = sorted(ratios)

    def pct(p):
        if not ratios_sorted:
            return None
        idx = int(round((len(ratios_sorted) - 1) * p))
        return ratios_sorted[idx]

    return {
        "n": len(ratios),
        "empty_docid_results": empty,
        "avg_prefix_match_ratio": (sum(ratios) / len(ratios)) if ratios else 0.0,
        "min": ratios_sorted[0] if ratios_sorted else None,
        "p25": pct(0.25),
        "median": pct(0.50),
        "p75": pct(0.75),
        "max": ratios_sorted[-1] if ratios_sorted else None,
    }


def main():
    for p in [
        "Results/test_hotpot_v11_200_results.json",
        "Results/test_hotpot_v11_200_results_v2.json",
    ]:
        res = load_results(p)
        stats = prefix_match_stats(res)
        print("\n" + p)
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
