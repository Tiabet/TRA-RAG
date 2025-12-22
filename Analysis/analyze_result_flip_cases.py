#!/usr/bin/env python3
"""Analyze answer flips between two result files (e.g., v4aligned vs v5).

We don't assume an LLM-eval file exists for both runs.
Instead, we use MRQA-style normalization to approximate correctness:
- exact match (normalized)
- accuracy containment (normalized gold in normalized pred)

Outputs:
- How many predictions changed
- Flip counts: A-correct/B-wrong and A-wrong/B-correct (by EM and by containment)
- A few representative examples

Usage:
  .venv/Scripts/python.exe Analysis/analyze_result_flip_cases.py \
    --a Results/test_musique_v11_200_results_v4aligned.json \
    --b Results/test_musique_v11_200_results_v5.json \
    --show 15
"""

from __future__ import annotations

import argparse
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def is_correct_em(gold: str, pred: str) -> bool:
    return normalize_answer(gold) == normalize_answer(pred)


def is_correct_contains(gold: str, pred: str) -> bool:
    ng = normalize_answer(gold)
    np = normalize_answer(pred)
    if not ng or not np:
        return False
    return ng in np


def load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    if isinstance(data, list):
        return data
    return []


def get_qid(item: Dict[str, Any], fallback_idx: int) -> str:
    return str(item.get('id') or item.get('_id') or f'idx{fallback_idx}')


def get_question(item: Dict[str, Any]) -> str:
    return str(item.get('question') or item.get('query') or '').strip()


def get_gold(item: Dict[str, Any]) -> str:
    return str(item.get('gold_answer') or item.get('answer') or '').strip()


def get_pred(item: Dict[str, Any]) -> str:
    v = item.get('predicted_answer')
    if v is None:
        v = item.get('final_answer')
    if v is None:
        return ''
    return str(v).strip()


@dataclass
class Row:
    qid: str
    question: str
    gold: str
    pred: str


def index_by_question(results: List[Dict[str, Any]]) -> Dict[str, Row]:
    out: Dict[str, Row] = {}
    for i, it in enumerate(results):
        q = get_question(it)
        if not q:
            continue
        key = re.sub(r"\s+", " ", q.strip())
        out[key] = Row(get_qid(it, i), q, get_gold(it), get_pred(it))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--show', type=int, default=10)
    args = ap.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)

    a_res = load_results(a_path)
    b_res = load_results(b_path)

    a_map = index_by_question(a_res)
    b_map = index_by_question(b_res)

    keys = sorted(set(a_map) & set(b_map))
    only_a = sorted(set(a_map) - set(b_map))
    only_b = sorted(set(b_map) - set(a_map))

    print('=' * 110)
    print(f'A: {a_path.as_posix()}  items={len(a_res)} unique_q={len(a_map)}')
    print(f'B: {b_path.as_posix()}  items={len(b_res)} unique_q={len(b_map)}')
    print(f'Overlap questions: {len(keys)}  only_A={len(only_a)} only_B={len(only_b)}')

    changed = 0
    flip_em_a_good = []  # (a good, b bad)
    flip_em_b_good = []
    flip_contain_a_good = []
    flip_contain_b_good = []

    a_em = 0
    b_em = 0
    a_cont = 0
    b_cont = 0

    for k in keys:
        a = a_map[k]
        b = b_map[k]

        if a.pred != b.pred:
            changed += 1

        a_em_ok = is_correct_em(a.gold, a.pred)
        b_em_ok = is_correct_em(b.gold, b.pred)
        a_ct_ok = is_correct_contains(a.gold, a.pred)
        b_ct_ok = is_correct_contains(b.gold, b.pred)

        a_em += int(a_em_ok)
        b_em += int(b_em_ok)
        a_cont += int(a_ct_ok)
        b_cont += int(b_ct_ok)

        if a_em_ok and not b_em_ok:
            flip_em_a_good.append((a, b))
        elif (not a_em_ok) and b_em_ok:
            flip_em_b_good.append((a, b))

        if a_ct_ok and not b_ct_ok:
            flip_contain_a_good.append((a, b))
        elif (not a_ct_ok) and b_ct_ok:
            flip_contain_b_good.append((a, b))

    n = len(keys)
    print('-' * 110)
    print(f'Changed predictions: {changed}/{n} ({(changed/max(1,n)):.3f})')
    print(f'EM accuracy:   A={a_em/n:.4f}  B={b_em/n:.4f}  delta={b_em/n - a_em/n:+.4f}')
    print(f'Containment:  A={a_cont/n:.4f}  B={b_cont/n:.4f}  delta={b_cont/n - a_cont/n:+.4f}')

    print('-' * 110)
    print(f'EM flips A-correct -> B-wrong: {len(flip_em_a_good)}')
    print(f'EM flips A-wrong -> B-correct: {len(flip_em_b_good)}')
    print(f'Contain flips A-correct -> B-wrong: {len(flip_contain_a_good)}')
    print(f'Contain flips A-wrong -> B-correct: {len(flip_contain_b_good)}')

    show_n = max(0, int(args.show))
    if show_n:
        def show(title: str, pairs: List[Tuple[Row, Row]]):
            print('-' * 110)
            print(title)
            for a, b in pairs[:show_n]:
                print(f'Q: {a.question}')
                print(f'  Gold: {a.gold}')
                print(f'  A: {a.pred}')
                print(f'  B: {b.pred}')
                print('')

        show('Examples (EM) A-correct -> B-wrong', flip_em_a_good)
        show('Examples (EM) A-wrong -> B-correct', flip_em_b_good)


if __name__ == '__main__':
    main()
