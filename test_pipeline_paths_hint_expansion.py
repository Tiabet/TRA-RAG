#!/usr/bin/env python3
"""Test runner for v12 paths-hint pipeline (final reranking, supports concurrency).

Examples:
    python test_pipeline_paths_hint_expansion.py --dataset musique --no_llm
    python test_pipeline_paths_hint_expansion.py --dataset hotpot --concurrency 10
    python test_pipeline_paths_hint_expansion.py --dataset musique --concurrency 20
"""

import argparse
import asyncio
import json
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from hybrid_path_retriever import HybridPathRetriever
from llm_logger import finalize_log, init_logger
from new_multihop_pipeline_paths_hint_expansion import (
    NewMultihopPipelineV12PathsHintExpansion,
    _default_artifact_paths,
    _json_default,
)


CONCURRENCY = 100


async def process_single_question(pipeline, item, idx: int, total: int, no_llm: bool, verbose: bool):
    question = item.get('question', '')
    gold_answer = item.get('answer')
    qid = item.get('_id') or item.get('id') or f'q{idx}'

    start = time.time()
    try:
        if no_llm:
            passages, paths = await pipeline.retrieve_for_query(question)
            result = {
                'success': True,
                'final_answer': gold_answer,
                'predicted_answer': gold_answer,
                'final_retrieved_passages': [
                    {
                        'doc_id': p.get('doc_id'),
                        'title': p.get('title'),
                        'passage_score': p.get('passage_score'),
                        'support_path_score': p.get('support_path_score'),
                        'support_path_origin': p.get('support_path_origin'),
                    }
                    for p in (passages or [])
                ],
                'final_retrieved_paths': [
                    {'doc_id': p.get('doc_id'), 'origin': p.get('origin'), 'score': p.get('score')}
                    for p in (paths or [])
                ],
                'decomposition': None,
                'num_passages': len(passages or []),
                'num_paths': len(paths or []),
                'time': 0.0,
            }
        else:
            result = await pipeline.process_question(question)

        elapsed = time.time() - start

        if result.get('success'):
            num_passages = result.get('num_passages', 0)
            num_paths = result.get('num_paths', None)
            paths_str = f", {num_paths}paths" if isinstance(num_paths, int) else ""
            print(f"[{idx+1:3d}/{total}] [OK] ({elapsed:.1f}s, {num_passages}p{paths_str}) {question[:60]}...")
            if verbose:
                print(f"           Gold: {gold_answer}")
                print(f"           Pred: {result.get('final_answer')}")
        else:
            err = str(result.get('error') or '')
            print(f"[{idx+1:3d}/{total}] [FAIL] ({elapsed:.1f}s) {err[:80]}...")

        merged = {
            'id': qid,
            'question': question,
            'gold_answer': gold_answer,
            'answer_aliases': item.get('answer_aliases', []),
        }
        merged.update(result)
        merged['time'] = float(result.get('time') or elapsed)
        if merged.get('predicted_answer') is None:
            merged['predicted_answer'] = merged.get('final_answer')
        return merged

    except Exception as e:
        elapsed = time.time() - start
        print(f"[{idx+1:3d}/{total}] [EXC] ({elapsed:.1f}s) {str(e)[:80]}...")
        return {
            'id': qid,
            'question': question,
            'gold_answer': gold_answer,
            'answer_aliases': item.get('answer_aliases', []),
            'predicted_answer': None,
            'success': False,
            'error': str(e),
            'time': elapsed,
        }


async def run_batch(pipeline, items, start_idx: int, total: int, no_llm: bool, verbose: bool):
    tasks = []
    for i, item in enumerate(items):
        idx = start_idx + i
        tasks.append(process_single_question(pipeline, item, idx, total, no_llm=no_llm, verbose=verbose))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            idx = start_idx + i
            item = items[i]
            processed.append(
                {
                    'id': item.get('_id') or item.get('id') or f'q{idx}',
                    'question': item.get('question', ''),
                    'gold_answer': item.get('answer'),
                    'predicted_answer': None,
                    'answer_aliases': item.get('answer_aliases', []),
                    'success': False,
                    'error': str(r),
                    'time': 0.0,
                }
            )
        else:
            processed.append(r)
    return processed


def parse_args():
    p = argparse.ArgumentParser(description='Test v12 paths-as-hints pipeline (final reranking)')
    p.add_argument('--dataset', choices=['musique', 'hotpot', '2wiki', 'lveval'], required=True)
    p.add_argument('--output', type=str, default='')
    # Optional artifact overrides (for robustness / combined-corpus runs)
    p.add_argument('--data_path', type=str, default='', help='Override QA JSON path (also used to build doc_id->passage map)')
    p.add_argument('--db_path', type=str, default='', help='Override SQLite DB path (metadata_v5.db)')
    p.add_argument('--bm25_index_path', type=str, default='', help='Override BM25 index directory (bm25_index_v5)')
    p.add_argument('--embeddings_path', type=str, default='', help='Override dense path embeddings (.npz)')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--no_llm', action='store_true')
    p.add_argument('--concurrency', type=int, default=CONCURRENCY)
    p.add_argument('--limit', type=int, default=0, help='Limit number of questions (0 = all)')
    p.add_argument('--top_k_passages', type=int, default=5, help='Answering passages (applies to SQ + Final)')
    p.add_argument('--top_k_paths', type=int, default=30, help='Answering paths (applies to SQ + Final)')
    p.add_argument('--bm25_weight', type=float, default=1.0)
    p.add_argument('--dense_weight', type=float, default=1.3)

    # Ablations
    p.add_argument(
        '--sq_fusion_method',
        type=str,
        default='rrf',
        choices=['rrf', 'minmax', 'bm25', 'dense'],
        help='Retriever fusion for SQ retrieval (rrf/minmax/bm25/dense)',
    )
    p.add_argument(
        '--final_selection_mode',
        type=str,
        default='rerank',
        choices=['rerank', 'rrf_only'],
        help='Final selection policy: rerank (default) or rrf_only (no rerank; SQ scores only)',
    )
    p.add_argument(
        '--no_previous_context',
        action='store_true',
        help='Disable previous-context injection (SQ dependency context + final subquestion chain)',
    )
    p.add_argument('--seed_k', type=int, default=20)
    p.add_argument('--expansion_k', type=int, default=10)
    p.add_argument('--expansion_dense_candidates', type=int, default=500)
    p.add_argument('--seed_passages_in_final', type=int, default=3)
    return p.parse_args()


async def run_test(args) -> None:
    load_dotenv()
    init_logger()

    from llm_provider import create_async_chat_client, detect_provider

    defaults = _default_artifact_paths(args.dataset)
    data_path = (str(args.data_path).strip() or str(defaults['data_path']))
    db_path = (str(args.db_path).strip() or str(defaults['db_path']))
    bm25_index_path = (str(args.bm25_index_path).strip() or str(defaults['bm25_index_path']))
    embeddings_path = (str(args.embeddings_path).strip() or str(defaults['embeddings_path']))

    output_path = args.output or f"Results/{args.dataset}_result.json"

    print("=" * 80)
    print(f"V12 Expansion Test - {str(args.dataset).upper()}")
    print("=" * 80)
    print(f"Concurrency: {int(args.concurrency)}")
    print(f"no_llm: {bool(args.no_llm)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    client = None
    if not args.no_llm:
        cfg = detect_provider()
        client = create_async_chat_client(cfg)

    retriever = HybridPathRetriever(
        bm25_index_path=bm25_index_path,
        embeddings_path=embeddings_path,
        bm25_weight=float(args.bm25_weight),
        dense_weight=float(args.dense_weight),
    )

    # Treat CLI top_k_* as *answering* counts (context size).
    # Keep internal retrieval pool large/stable so ranking signals remain even when answer_k_paths=0.
    # This matches the intent of "answering counts can be 0" while keeping retrieval + scoring enabled.
    answer_k_passages = int(args.top_k_passages)
    answer_k_paths = int(args.top_k_paths)
    retrieval_top_k_passages = max(20, answer_k_passages)
    retrieval_top_k_paths = max(50, answer_k_paths)
    print(
        f"Answering counts: passages={answer_k_passages}, paths={answer_k_paths} | "
        f"Retrieval pool: passages={retrieval_top_k_passages}, paths={retrieval_top_k_paths}"
    )

    pipeline = NewMultihopPipelineV12PathsHintExpansion(
        client=client,  # type: ignore[arg-type]
        retriever=retriever,
        hotpotqa_path=data_path,
        db_path=db_path,
        top_k_passages=int(retrieval_top_k_passages),
        top_k_paths=int(retrieval_top_k_paths),
        answer_k_passages=int(answer_k_passages),
        answer_k_paths=int(answer_k_paths),
        path_fetch_k=50,
        verbose=bool(args.verbose),
        seed_k=int(args.seed_k),
        expansion_k=int(args.expansion_k),
        expansion_dense_candidates=int(args.expansion_dense_candidates),
        seed_passages_in_final=int(args.seed_passages_in_final),
        final_rerank_mode='minmax',
		sq_fusion_method=str(getattr(args, 'sq_fusion_method', 'rrf')),
		final_selection_mode=str(getattr(args, 'final_selection_mode', 'rerank')),
		use_previous_context=(not bool(getattr(args, 'no_previous_context', False))),
    )

    print(f"\nLoading data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Sanity check: the pipeline runner expects a QA-style JSON list.
    # Some corpora (e.g., 2WikiMultihopQA/2wikimultihopqa.json in this repo) contain only paragraphs.
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list of QA items, got: {type(data)} ({data_path})")

    has_any_question = any(
        isinstance(it, dict) and isinstance(it.get('question'), str) and it.get('question').strip()
        for it in data
    )
    if not has_any_question:
        raise SystemExit(
            "Input file does not contain any non-empty 'question' fields. "
            "This usually means you're pointing at a corpus-only JSON (paragraphs only), not a QA dataset. "
            f"Please pass a QA JSON with at least {{question, answer}} fields. (data_path={data_path})"
        )

    if int(getattr(args, 'limit', 0) or 0) > 0:
        data = data[: int(args.limit)]
    total = len(data)

    # Build a stable fingerprint of the exact subset being evaluated.
    subset_ids = [
        str((it or {}).get('_id') or (it or {}).get('id') or f'q{i}')
        for i, it in enumerate(data)
        if isinstance(it, dict)
    ]
    subset_hash = hashlib.sha1('\n'.join(subset_ids).encode('utf-8')).hexdigest() if subset_ids else ''
    limit_note = f" (limit={int(args.limit)})" if int(getattr(args, 'limit', 0) or 0) > 0 else ""
    if subset_hash:
        print(f"[OK] Loaded {total} questions{limit_note} | subset_sha1={subset_hash[:12]}")
    else:
        print(f"[OK] Loaded {total} questions{limit_note}")
    Path('Results').mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    all_results = []

    def write_snapshot(results_list, is_final: bool):
        elapsed_total = time.time() - start_time
        success_count = sum(1 for r in results_list if r.get('success'))
        error_count = len(results_list) - success_count

        successful_results = [r for r in results_list if r.get('success')]
        avg_time = (
            sum(float(r.get('time') or 0.0) for r in successful_results) / len(successful_results)
            if successful_results
            else 0.0
        )

        output = {
            'config': {
                'pipeline_version': 'v12_final_rerank',
                'dataset': str(args.dataset),
                'concurrency': int(args.concurrency),
                'limit': int(getattr(args, 'limit', 0) or 0),
                'subset_sha1': str(subset_hash),
                # Final selection policy:
                # - Answering totals are controlled by answer_k_passages/answer_k_paths
                # - Seed/rerank splits come from v12 mapping tables (see pipeline code)
                'seed_k': int(args.seed_k),
                'expansion_k': int(args.expansion_k),
                'expansion_dense_candidates': int(args.expansion_dense_candidates),
                'seed_passages_in_final': int(args.seed_passages_in_final),
                'final_rerank_mode': 'minmax',
                'top_k_passages': int(retrieval_top_k_passages),
                'top_k_paths': int(retrieval_top_k_paths),
                'answer_k_passages': int(answer_k_passages),
                'answer_k_paths': int(answer_k_paths),
                'path_fetch_k_input': 50,
                'bm25_weight': float(args.bm25_weight),
                'dense_weight': float(args.dense_weight),
                'sq_fusion_method': str(getattr(args, 'sq_fusion_method', 'rrf')),
                'final_selection_mode': str(getattr(args, 'final_selection_mode', 'rerank')),
                'use_previous_context': (not bool(getattr(args, 'no_previous_context', False))),
                'dense_enabled': bool(getattr(retriever, 'embed_client', None)),
                'no_llm': bool(args.no_llm),
                'artifacts': {
                    'data_path': data_path,
                    'db_path': db_path,
                    'bm25_index_path': bm25_index_path,
                    'embeddings_path': embeddings_path,
                },
            },
            'summary': {
                'total_questions': total,
                'completed_questions': len(results_list),
                'successful': success_count,
                'errors': error_count,
                'total_time': elapsed_total,
                'avg_time_per_question': avg_time,
                'timestamp': datetime.now().isoformat(),
                'is_final': bool(is_final),
            },
            'results': results_list,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=_json_default)

    try:
        concurrency = max(1, int(args.concurrency))
        for batch_start in range(0, total, concurrency):
            batch_end = min(batch_start + concurrency, total)
            batch_items = data[batch_start:batch_end]
            print(f"\n>>> Batch {batch_start//concurrency + 1}: Questions {batch_start+1}-{batch_end}")

            batch_results = await run_batch(
                pipeline,
                batch_items,
                start_idx=batch_start,
                total=total,
                no_llm=bool(args.no_llm),
                verbose=bool(args.verbose),
            )
            all_results.extend(batch_results)

            elapsed = time.time() - start_time
            success_count = sum(1 for r in all_results if r.get('success'))
            error_count = len(all_results) - success_count
            print(
                f"Completed: {len(all_results)}/{total} | Success: {success_count} | Errors: {error_count} | Time: {elapsed:.0f}s"
            )
            write_snapshot(all_results, is_final=False)

    except KeyboardInterrupt:
        print("\n[Interrupted] Writing partial results...")
        write_snapshot(all_results, is_final=False)
        pipeline.close()
        raise

    pipeline.close()
    log_path = finalize_log()
    print(f"[LOG] {log_path}")
    write_snapshot(all_results, is_final=True)
    print(f"[OK] Wrote: {output_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_test(args))


if __name__ == '__main__':
    main()
