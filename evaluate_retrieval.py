import json
import argparse
import numpy as np
from typing import Set, List, Dict, Tuple

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict) and 'results' in data:
            return data['results']
        return data

def get_gold_titles(item) -> Set[str]:
    """Extract unique titles from supporting_facts."""
    return set(fact[0] for fact in item.get('supporting_facts', []))


def _build_title_to_doc_ids(item: Dict) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    sample_id = item.get('_id') or item.get('id')

    # MuSiQue-style: paragraphs already have corpus_idx
    if isinstance(item.get('paragraphs'), list):
        for p in item.get('paragraphs') or []:
            if not isinstance(p, dict):
                continue
            title = p.get('title')
            corpus_idx = p.get('corpus_idx')
            if title is None:
                continue
            if corpus_idx is not None:
                mapping.setdefault(str(title), []).append(str(corpus_idx))
        return mapping

    # HotpotQA-style: context can be list-based or dict-based
    for ctx_idx, c in enumerate(item.get('context', []) or []):
        if isinstance(c, list) and len(c) >= 1:
            title = c[0]
            doc_id = f"{sample_id}::ctx{ctx_idx}" if sample_id is not None else f"ctx{ctx_idx}"
            mapping.setdefault(str(title), []).append(str(doc_id))
        elif isinstance(c, dict):
            title = c.get('title')
            if title is None:
                continue
            corpus_idx = c.get('corpus_idx')
            if corpus_idx is not None:
                doc_id = str(corpus_idx)
            else:
                local_idx = c.get('local_idx', ctx_idx)
                doc_id = f"{sample_id}::ctx{int(local_idx)}" if sample_id is not None else f"ctx{int(local_idx)}"
            mapping.setdefault(str(title), []).append(str(doc_id))
    return mapping


def get_gold_doc_ids(item: Dict) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (gold_doc_ids, missing_titles, ambiguous_titles).

    supporting_facts provides titles; we map them to doc_ids via the item's context.
    - missing_titles: titles in supporting_facts that are not present in context
    - ambiguous_titles: titles that map to multiple ctx entries for this item
    """
    title_to_doc_ids = _build_title_to_doc_ids(item)
    gold_doc_ids: Set[str] = set()
    missing_titles: Set[str] = set()
    ambiguous_titles: Set[str] = set()

    for title, _sent_idx in item.get('supporting_facts', []):
        doc_ids = title_to_doc_ids.get(title)
        if not doc_ids:
            missing_titles.add(title)
            continue
        if len(doc_ids) > 1:
            ambiguous_titles.add(title)
        gold_doc_ids.update(doc_ids)

    return gold_doc_ids, missing_titles, ambiguous_titles


def get_gold_corpus_doc_ids(item: Dict) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (gold_corpus_doc_ids, missing_titles, ambiguous_titles) using corpus_idx IDs.

    - MuSiQue: paragraphs[].is_supporting == True => use paragraphs[].corpus_idx
    - HotpotQA: supporting_facts titles => map to context[].corpus_idx (can be many)

    If corpus_idx fields are not found, falls back to legacy get_gold_doc_ids().
    """

    # MuSiQue-style
    if isinstance(item.get('paragraphs'), list):
        gold: Set[str] = set()
        for p in item.get('paragraphs') or []:
            if not isinstance(p, dict):
                continue
            if not p.get('is_supporting'):
                continue
            corpus_idx = p.get('corpus_idx')
            if corpus_idx is None:
                continue
            gold.add(str(corpus_idx))
        return gold, set(), set()

    # HotpotQA-style: context dict with corpus_idx
    has_any_corpus_idx = False
    title_to_ids: Dict[str, List[str]] = {}
    for c in item.get('context', []) or []:
        if isinstance(c, dict):
            title = c.get('title')
            corpus_idx = c.get('corpus_idx')
            if title is None or corpus_idx is None:
                continue
            has_any_corpus_idx = True
            title_to_ids.setdefault(str(title), []).append(str(corpus_idx))

    if not has_any_corpus_idx:
        return get_gold_doc_ids(item)

    gold_doc_ids: Set[str] = set()
    missing_titles: Set[str] = set()
    ambiguous_titles: Set[str] = set()
    for title, _sent_idx in item.get('supporting_facts', []) or []:
        ids = title_to_ids.get(str(title))
        if not ids:
            missing_titles.add(str(title))
            continue
        if len(ids) > 1:
            ambiguous_titles.add(str(title))
        gold_doc_ids.update(ids)

    return gold_doc_ids, missing_titles, ambiguous_titles

def get_retrieved_titles(result_item) -> Set[str]:
    """Extract unique titles from retrieved passages in the result."""
    titles = set()
    
    # 1. Check top-level retrieved_passages (for No-QD pipelines)
    if 'retrieved_passages' in result_item:
        passages = result_item['retrieved_passages']
        for p in passages:
            if isinstance(p, dict):
                titles.add(p.get('title', ''))
            elif isinstance(p, str):
                titles.add(p)
    
    # 1.1 Check retrieved_docs (simple list of titles)
    if 'retrieved_docs' in result_item:
        for t in result_item['retrieved_docs']:
            if isinstance(t, str):
                titles.add(t)

    # 2. Check decomposition for retrieved passages (for QD pipelines)
    decomposition = result_item.get('decomposition')
    if decomposition:
        subquestions = []
        if isinstance(decomposition, list):
            subquestions = decomposition
        elif isinstance(decomposition, dict):
            subquestions = decomposition.get('subquestions', [])
            
        for sq in subquestions:
            passages = sq.get('retrieved_passages', [])
            for p in passages:
                if isinstance(p, dict):
                    titles.add(p.get('title', ''))
                elif isinstance(p, str):
                    titles.add(p)
                    
    return titles


def get_retrieved_titles_resolved_from_doc_id(result_item: Dict, doc_id_to_title: Dict[str, str]) -> Set[str]:
    """Extract unique titles, preferring doc_id→title resolution when doc_id is present."""
    titles: Set[str] = set()

    def _add_title_from_passage(p: Dict):
        doc_id = p.get('doc_id')
        if doc_id is not None:
            resolved = doc_id_to_title.get(str(doc_id))
            if resolved:
                titles.add(resolved)
                return
        t = p.get('title', '')
        if isinstance(t, str) and t:
            titles.add(t)

    # 1) Top-level retrieved_passages
    for p in result_item.get('retrieved_passages', []) or []:
        if isinstance(p, dict):
            _add_title_from_passage(p)
        elif isinstance(p, str):
            titles.add(p)

    # 1.1) retrieved_docs (simple list of titles)
    for t in result_item.get('retrieved_docs', []) or []:
        if isinstance(t, str) and t:
            titles.add(t)

    # 2) Decomposition subquestions
    decomposition = result_item.get('decomposition')
    subquestions = []
    if isinstance(decomposition, list):
        subquestions = decomposition
    elif isinstance(decomposition, dict):
        subquestions = decomposition.get('subquestions', []) or []

    for sq in subquestions:
        for p in sq.get('retrieved_passages', []) or []:
            if isinstance(p, dict):
                _add_title_from_passage(p)
            elif isinstance(p, str):
                titles.add(p)

    return titles


def get_retrieved_doc_ids(result_item: Dict, sources: str = 'both') -> Set[str]:
    """Extract unique doc_ids from retrieved passages/paths in the result.

    sources: 'passages' | 'paths' | 'both'
    """
    doc_ids: Set[str] = set()

    def _add_doc_id(value):
        if not value:
            return
        if isinstance(value, str):
            doc_ids.add(value)
        elif isinstance(value, (int, float)):
            doc_ids.add(str(value))

    if sources in ('passages', 'both'):
        # 1) Top-level retrieved_passages
        for p in result_item.get('retrieved_passages', []) or []:
            if isinstance(p, dict):
                if 'doc_id' in p:
                    _add_doc_id(p.get('doc_id'))
                if 'doc_ids' in p and isinstance(p.get('doc_ids'), list):
                    for d in p.get('doc_ids'):
                        _add_doc_id(d)

    # 2) Decomposition subquestions
    decomposition = result_item.get('decomposition')
    subquestions = []
    if isinstance(decomposition, list):
        subquestions = decomposition
    elif isinstance(decomposition, dict):
        subquestions = decomposition.get('subquestions', []) or []

    for sq in subquestions:
        if sources in ('passages', 'both'):
            for p in sq.get('retrieved_passages', []) or []:
                if isinstance(p, dict):
                    if 'doc_id' in p:
                        _add_doc_id(p.get('doc_id'))
                    if 'doc_ids' in p and isinstance(p.get('doc_ids'), list):
                        for d in p.get('doc_ids'):
                            _add_doc_id(d)
        if sources in ('paths', 'both'):
            for rp in sq.get('retrieved_paths', []) or []:
                if isinstance(rp, dict) and 'doc_id' in rp:
                    _add_doc_id(rp.get('doc_id'))

    if sources in ('paths', 'both'):
        # 3) Some pipelines store paths at top level
        for rp in result_item.get('retrieved_paths', []) or []:
            if isinstance(rp, dict) and 'doc_id' in rp:
                _add_doc_id(rp.get('doc_id'))

    return doc_ids


def get_final_retrieved_doc_ids_at_k(result_item: Dict, k: int = 5, sources: str = 'passages') -> Set[str]:
    """Extract doc_ids for final-answer retrieval only, truncated to @k.

    Expected fields (emitted by v11 pipelines):
      - final_retrieved_passages: List[{doc_id, title, ...}] (ordered)
      - final_retrieved_paths: List[{doc_id, ...}] (optional)

    sources: 'passages' | 'paths' | 'both'
    """
    doc_ids: List[str] = []

    def _safe_score(p: Dict) -> float:
        try:
            s = p.get('score', None)
            return float(s) if s is not None else float('-inf')
        except Exception:
            return float('-inf')

    def _path_dedupe_key(p: Dict) -> Tuple[str, str, str, str]:
        source_title = p.get('source_title') or p.get('title') or ''
        entity_title = p.get('entity_title') or p.get('title') or ''
        key_path = p.get('key_path', '')
        value = p.get('value', '')
        return (str(source_title), str(entity_title), str(key_path), str(value))

    def _push(value):
        if not value:
            return
        if isinstance(value, str):
            doc_ids.append(value)
        elif isinstance(value, (int, float)):
            doc_ids.append(str(value))

    if sources in ('passages', 'both'):
        for p in (result_item.get('final_retrieved_passages') or []):
            if isinstance(p, dict):
                _push(p.get('doc_id'))
            else:
                _push(p)

    if sources in ('paths', 'both'):
        for p in (result_item.get('final_retrieved_paths') or []):
            if isinstance(p, dict):
                _push(p.get('doc_id'))
            else:
                _push(p)

    # Fallback for older result files: derive final doc_ids from decomposition.retrieved_paths.
    # This matches the pipeline's final selection logic:
    #   - collect UNIQUE paths across SQs
    #   - sort by score desc
    #   - take top-30
    #   - choose first k unique doc_ids
    if not doc_ids:
        decomposition = result_item.get('decomposition') or {}
        subqs = []
        if isinstance(decomposition, dict):
            subqs = decomposition.get('subquestions', []) or []
        elif isinstance(decomposition, list):
            subqs = decomposition

        seen_path = set()
        unique_paths: List[Dict] = []
        for sq in subqs:
            for p in (sq.get('retrieved_paths') or []):
                if not isinstance(p, dict):
                    continue
                key = _path_dedupe_key(p)
                if key in seen_path:
                    continue
                seen_path.add(key)
                unique_paths.append(p)

        unique_paths_sorted = sorted(unique_paths, key=_safe_score, reverse=True)
        top_paths = unique_paths_sorted[:30]
        for p in top_paths:
            # Final passages are selected by doc_id from high-score paths,
            # so doc_id_sources='passages' can safely use this fallback too.
            if sources not in ('passages', 'paths', 'both'):
                break
            if isinstance(p, dict):
                _push(p.get('doc_id'))

    # Preserve order but enforce uniqueness
    seen = set()
    ordered_unique = []
    for d in doc_ids:
        ds = str(d)
        if not ds or ds in seen:
            continue
        seen.add(ds)
        ordered_unique.append(ds)
        if len(ordered_unique) >= k:
            break

    return set(ordered_unique)


def _analyze_title_uniqueness(gold_data: List[Dict]) -> Dict[str, int]:
    """Quick diagnostics to surface title ambiguity issues (esp. MuSiQue vs HotpotQA)."""
    per_item_ambiguous = 0
    total_items = 0

    global_title_counts: Dict[str, int] = {}
    for item in gold_data:
        total_items += 1
        title_to_doc_ids = _build_title_to_doc_ids(item)
        if any(len(v) > 1 for v in title_to_doc_ids.values()):
            per_item_ambiguous += 1
        for title in title_to_doc_ids.keys():
            global_title_counts[title] = global_title_counts.get(title, 0) + 1

    global_duplicate_titles = sum(1 for _t, c in global_title_counts.items() if c > 1)
    return {
        "items": total_items,
        "items_with_ambiguous_title_in_context": per_item_ambiguous,
        "global_unique_titles": len(global_title_counts),
        "global_duplicate_titles": global_duplicate_titles,
    }

def evaluate(
    result_path,
    gold_path,
    key: str = 'doc_id',
    check_mapping: bool = False,
    resolve_titles_from_doc_id: bool = False,
    doc_id_sources: str = 'both',
    scope: str = 'final',
    at_k: int = 5,
    gold_id: str = 'doc_id',
):
    results = load_json(result_path)
    gold_data = load_json(gold_path)

    print("=" * 90)
    print("[evaluate_retrieval] Running evaluation")
    print(f"result_path: {result_path}")
    print(f"gold_path:   {gold_path}")
    print(f"key:         {key}")
    if key == 'doc_id':
        print(f"gold_id:     {gold_id}")
    print(f"scope:       {scope}{'@'+str(at_k) if (key=='doc_id' and scope=='final') else ''}")
    print(f"doc_id_sources: {doc_id_sources}")
    print("=" * 90)
    
    # Create a map for gold data
    gold_map = {item['_id']: item for item in gold_data}
    # Also map by question text as fallback if ID is missing or different
    gold_q_map = {item['question']: item for item in gold_data}
    
    metrics = {
        'recall': [],
        'precision': [],
        'f1': [],
        'hit_rate': [] # 1 if recall == 1.0 else 0 (All gold passages retrieved)
    }
    
    print(f"Evaluating {len(results)} results...")

    if check_mapping:
        stats = _analyze_title_uniqueness(gold_data)
        print("\n[Gold title/context diagnostics]")
        print(f"Items: {stats['items']}")
        print(f"Items with ambiguous title in context: {stats['items_with_ambiguous_title_in_context']}")
        print(f"Global unique titles: {stats['global_unique_titles']}")
        print(f"Global duplicate titles (across items): {stats['global_duplicate_titles']}")

    gold_doc_id_universe: Set[str] = set()
    doc_id_to_title: Dict[str, str] = {}
    if key == 'doc_id':
        if gold_id == 'corpus_idx':
            for item in gold_data:
                # MuSiQue paragraphs
                if isinstance(item.get('paragraphs'), list):
                    for p in item.get('paragraphs') or []:
                        if not isinstance(p, dict):
                            continue
                        corpus_idx = p.get('corpus_idx')
                        title = p.get('title')
                        if corpus_idx is None:
                            continue
                        gold_doc_id_universe.add(str(corpus_idx))
                        if title is not None:
                            doc_id_to_title.setdefault(str(corpus_idx), str(title))
                    continue

                # HotpotQA context dict
                for c in item.get('context', []) or []:
                    if not isinstance(c, dict):
                        continue
                    corpus_idx = c.get('corpus_idx')
                    title = c.get('title')
                    if corpus_idx is None:
                        continue
                    gold_doc_id_universe.add(str(corpus_idx))
                    if title is not None:
                        doc_id_to_title.setdefault(str(corpus_idx), str(title))
        else:
            for item in gold_data:
                sample_id = item.get('_id')
                if not sample_id:
                    continue
                for ctx_idx, _ctx in enumerate(item.get('context', [])):
                    gold_doc_id_universe.add(f"{sample_id}::ctx{ctx_idx}")

    # Useful for resolving titles from doc_id when comparing legacy result files.
    if not doc_id_to_title:
        for item in gold_data:
            sample_id = item.get('_id')
            if not sample_id:
                continue
            for ctx_idx, c in enumerate(item.get('context', []) or []):
                if isinstance(c, list) and len(c) >= 1:
                    title = c[0]
                    doc_id_to_title[f"{sample_id}::ctx{ctx_idx}"] = str(title)

    missing_gold_items = 0
    missing_titles_total = 0
    ambiguous_titles_total = 0
    retrieved_doc_ids_total = 0
    retrieved_doc_ids_in_gold_total = 0
    missing_final_fields = 0
    
    count = 0
    for res in results:
        question = res['question']
        
        # Find corresponding gold item
        gold_item = gold_q_map.get(question)
        if not gold_item:
            # Try finding by ID if available
            if '_id' in res and res['_id'] in gold_map:
                gold_item = gold_map[res['_id']]
        
        if not gold_item:
            # print(f"Warning: Gold item not found for question: {question[:50]}...")
            continue
            
        if key == 'title':
            gold_set = get_gold_titles(gold_item)
            if resolve_titles_from_doc_id:
                retrieved_set = get_retrieved_titles_resolved_from_doc_id(res, doc_id_to_title)
            else:
                retrieved_set = get_retrieved_titles(res)
        else:
            if gold_id == 'corpus_idx':
                gold_set, missing_titles, ambiguous_titles = get_gold_corpus_doc_ids(gold_item)
            else:
                gold_set, missing_titles, ambiguous_titles = get_gold_doc_ids(gold_item)
            missing_titles_total += len(missing_titles)
            ambiguous_titles_total += len(ambiguous_titles)

            if scope == 'final':
                if (res.get('final_retrieved_passages') is None) and (res.get('final_retrieved_paths') is None):
                    missing_final_fields += 1
                retrieved_set = get_final_retrieved_doc_ids_at_k(res, k=at_k, sources=doc_id_sources)
            else:
                retrieved_set = get_retrieved_doc_ids(res, sources=doc_id_sources)

            retrieved_doc_ids_total += len(retrieved_set)
            if gold_doc_id_universe:
                retrieved_doc_ids_in_gold_total += len(retrieved_set.intersection(gold_doc_id_universe))

        # Calculate metrics
        intersection = gold_set.intersection(retrieved_set)

        recall = len(intersection) / len(gold_set) if gold_set else 0
        precision = len(intersection) / len(retrieved_set) if retrieved_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['recall'].append(recall)
        metrics['precision'].append(precision)
        metrics['f1'].append(f1)
        metrics['hit_rate'].append(1.0 if recall == 1.0 else 0.0)
        
        count += 1
        
    scope_label = f"{scope}@{at_k}" if (key == 'doc_id' and scope == 'final') else scope
    print(f"\nEvaluation Results ({count} questions, key={key}, scope={scope_label}):")
    metric_suffix = f"@{at_k}" if (key == 'doc_id' and scope == 'final') else ""
    print(f"Avg Recall{metric_suffix}:    {np.mean(metrics['recall']):.4f}")
    print(f"Avg Precision{metric_suffix}: {np.mean(metrics['precision']):.4f}")
    print(f"Avg F1{metric_suffix}:        {np.mean(metrics['f1']):.4f}")
    print(f"Hit Rate (All){metric_suffix}: {np.mean(metrics['hit_rate']):.4f}")

    if key == 'doc_id':
        print("\n[DocID mapping diagnostics]")
        print(f"Total missing supporting_facts titles in context: {missing_titles_total}")
        print(f"Total ambiguous supporting_facts titles in context: {ambiguous_titles_total}")
        if scope == 'final':
            print(f"Results missing final_retrieved_* fields: {missing_final_fields}")
            if len(results) > 0 and missing_final_fields == len(results):
                print("\n[WARNING] All results are missing `final_retrieved_passages`/`final_retrieved_paths`.")
                print("This means scope=final@k is using the fallback reconstruction from `decomposition.subquestions[].retrieved_paths`.")
                print("If you intended *true* final@k (final answer uses exactly these k passages), re-run evaluation with a newer result file")
                print("that contains `final_retrieved_passages` and/or `final_retrieved_paths`.")
                print("Example:")
                print("  python evaluate_retrieval.py --result_path Results/<new_results_with_final_fields>.json --gold_path <gold>.json --scope final --at_k 5")
        if retrieved_doc_ids_total > 0 and gold_doc_id_universe:
            coverage = retrieved_doc_ids_in_gold_total / retrieved_doc_ids_total
            print(f"Retrieved doc_ids that exist in gold context universe: {coverage:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default ='Results/test_musique_v12_ragcot_results_v4aligned_v1.json', help='Path to result JSON file')
    parser.add_argument('--gold_path', type=str, default='MuSiQue/musique_sample_200.json', help='Path to gold dataset')
    # parser.add_argument('--gold_path', type=str, default='HotpotQA/hotpotqa_sample_200.json', help='Path to gold dataset')
    parser.add_argument('--key', type=str, default='doc_id', choices=['doc_id', 'title'], help='Evaluation key')
    parser.add_argument('--check_mapping', action='store_true', help='Print title/doc_id mapping diagnostics')
    parser.add_argument('--resolve_titles_from_doc_id', action='store_true', help='When --key title, resolve titles using doc_id->title mapping from gold dataset')
    parser.add_argument('--doc_id_sources', type=str, default='passages', choices=['passages', 'paths', 'both'], help='When --key doc_id, which result fields to count')
    parser.add_argument('--scope', type=str, default='final', choices=['final', 'all'], help='Evaluate only final@k retrieval or union over all sub-questions')
    parser.add_argument('--at_k', type=int, default=5, help='k for final@k evaluation (only when --scope final and --key doc_id)')
    parser.add_argument('--gold_id', type=str, default='doc_id', choices=['doc_id', 'corpus_idx'], help='Gold doc_id scheme: legacy qid::ctxN or corpus_idx')
    args = parser.parse_args()

    evaluate(
        args.result_path,
        args.gold_path,
        key=args.key,
        check_mapping=args.check_mapping,
        resolve_titles_from_doc_id=bool(args.resolve_titles_from_doc_id),
        doc_id_sources=str(args.doc_id_sources),
        scope=str(args.scope),
        at_k=int(args.at_k),
        gold_id=str(args.gold_id),
    )