#!/usr/bin/env python3
"""Find an example where SQ retrieved_passages doc_id is not present in SQ retrieved_paths doc_ids.

This should normally never happen if passages are derived from paths by doc_id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def load_results(path: Path) -> List[Dict[str, Any]]:
    data = json.load(path.open('r', encoding='utf-8'))
    if isinstance(data, dict) and 'results' in data:
        data = data['results']
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def doc_ids_from_paths(paths: Any) -> Set[str]:
    out: Set[str] = set()
    for p in paths or []:
        if isinstance(p, dict) and p.get('doc_id') is not None:
            out.add(str(p.get('doc_id')))
    return out


def doc_ids_from_passages(passages: Any) -> List[str]:
    out: List[str] = []
    for p in passages or []:
        if isinstance(p, dict) and p.get('doc_id') is not None:
            out.append(str(p.get('doc_id')))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_json', required=True)
    ap.add_argument('--max_examples', type=int, default=1)
    args = ap.parse_args()

    results = load_results(Path(args.result_json))
    shown = 0

    for item in results:
        decomp = item.get('decomposition')
        subqs = []
        if isinstance(decomp, dict):
            subqs = decomp.get('subquestions', []) or []
        elif isinstance(decomp, list):
            subqs = decomp

        for sq in subqs:
            if not isinstance(sq, dict):
                continue
            raw_paths = sq.get('retrieved_paths')
            sq_paths = doc_ids_from_paths(raw_paths)
            if not sq_paths:
                continue
            sq_pass = doc_ids_from_passages(sq.get('retrieved_passages'))
            for d in sq_pass:
                if d not in sq_paths:
                    print('=== MISMATCH EXAMPLE ===')
                    print('question_id:', item.get('id'))
                    print('question:', item.get('question'))
                    print('sq_id:', sq.get('id'))
                    print('sq_question:', sq.get('question'))
                    print('missing passage doc_id:', d)
                    print('sq_paths_count:', len(sq_paths))
                    print('sq_passages_count:', len(sq_pass))
                    missing_doc_id_paths = 0
                    for p in raw_paths or []:
                        if isinstance(p, dict) and p.get('doc_id') in (None, ''):
                            missing_doc_id_paths += 1
                    print('sq_paths_missing_doc_id_entries:', missing_doc_id_paths)
                    # show a few path doc_ids and passage doc_ids
                    print('sample sq_paths doc_ids:', list(sorted(sq_paths))[:10])
                    print('sq_passage doc_ids:', sq_pass)
                    # show first few raw paths for inspection
                    print('sample raw retrieved_paths entries (first 3):')
                    for p in (raw_paths or [])[:3]:
                        print('  -', {k: p.get(k) for k in ['doc_id','source_title','entity_title','key_path','value','score'] if isinstance(p, dict)})
                    shown += 1
                    break
            if shown >= args.max_examples:
                return 0

    print('No mismatches found (or no SQ paths recorded).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
