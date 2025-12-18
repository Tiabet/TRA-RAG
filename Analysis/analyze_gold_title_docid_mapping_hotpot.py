#!/usr/bin/env python3
"""Analyze HotpotQA gold title→doc_id mapping correctness/ambiguity.

This directly answers: "retrieval 평가 시 gold title로부터 doc_id를 잘 찾아내고 있느냐".

HotpotQA gold supervision is in supporting_facts (titles). We map those titles to
context entries and thus to doc_ids of the form {id}::ctx{idx}.

Outputs:
- missing: supporting_facts title not found in that item's context
- ambiguous: supporting_facts title appears multiple times in that item's context
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def build_title_to_doc_ids(item: Dict) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    sample_id = item.get('_id')
    if not sample_id:
        return mapping
    for ctx_idx, (title, _sentences) in enumerate(item.get('context', []) or []):
        mapping.setdefault(str(title), []).append(f"{sample_id}::ctx{ctx_idx}")
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold_json', required=True, help='HotpotQA/hotpotqa_sample_200.json')
    ap.add_argument('--max_examples', type=int, default=10)
    args = ap.parse_args()

    gold = load_json(Path(args.gold_json))

    missing_total = 0
    ambiguous_total = 0
    items_with_missing = 0
    items_with_ambiguous = 0

    missing_examples: List[Tuple[str, str]] = []
    ambiguous_examples: List[Tuple[str, str, int]] = []
    ambiguous_title_counts: Counter[str] = Counter()

    for item in gold:
        if not isinstance(item, dict):
            continue
        qid = item.get('_id')
        if not qid:
            continue

        title_to_doc_ids = build_title_to_doc_ids(item)
        missing_here: Set[str] = set()
        ambiguous_here: Set[str] = set()

        for title, _sent_idx in item.get('supporting_facts', []) or []:
            title = str(title)
            doc_ids = title_to_doc_ids.get(title)
            if not doc_ids:
                missing_here.add(title)
                continue
            if len(doc_ids) > 1:
                ambiguous_here.add(title)
                ambiguous_title_counts[title] += 1

        if missing_here:
            items_with_missing += 1
            for t in list(missing_here)[: max(0, args.max_examples - len(missing_examples))]:
                missing_examples.append((qid, t))
            missing_total += len(missing_here)

        if ambiguous_here:
            items_with_ambiguous += 1
            for t in list(ambiguous_here)[: max(0, args.max_examples - len(ambiguous_examples))]:
                ambiguous_examples.append((qid, t, len(title_to_doc_ids.get(t, []))))
            ambiguous_total += len(ambiguous_here)

    print('\n[HotpotQA supporting_facts title→doc_id mapping]')
    print(f'Items: {len(gold)}')
    print(f'Items with missing titles: {items_with_missing}')
    print(f'Total missing titles (unique per item): {missing_total}')
    print(f'Items with ambiguous titles: {items_with_ambiguous}')
    print(f'Total ambiguous titles (unique per item): {ambiguous_total}')

    if missing_examples:
        print('\nExamples (missing):')
        for qid, t in missing_examples[: args.max_examples]:
            print(f'  - {qid}: {t}')

    if ambiguous_examples:
        print('\nExamples (ambiguous):')
        for qid, t, n in ambiguous_examples[: args.max_examples]:
            print(f'  - {qid}: {t} (maps to {n} doc_ids)')

    if ambiguous_title_counts:
        most = ambiguous_title_counts.most_common(10)
        print('\nTop ambiguous supporting_facts titles (by frequency across items):')
        for t, c in most:
            print(f'  - {t}: {c}')

    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
