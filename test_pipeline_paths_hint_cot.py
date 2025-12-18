#!/usr/bin/env python3
"""\
Test Dataset Pipeline (Paths-as-Hints + rag_qa_cot one-shot)
==========================================================

Runs either MuSiQue or HotpotQA pipeline on sample questions.
This is an integration-style test (requires network + API credentials).

Usage:
  python test_pipeline_paths_hint_cot.py
  python test_pipeline_paths_hint_cot.py --dataset hotpot
  python test_pipeline_paths_hint_cot.py --max_questions 5
  python test_pipeline_paths_hint_cot.py --concurrency 10
  python test_pipeline_paths_hint_cot.py --output Results/test_musique_v12_ragprompt_results_v4aligned.json
  python test_pipeline_paths_hint_cot.py --dataset hotpot --output Results/test_hotpot_v12_ragcot_results_v4aligned.json
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from hybrid_path_retriever import HybridPathRetriever
from new_multihop_pipeline_paths_hint_cot import NewMultihopPipelineV11PathsHintCoT
from llm_logger import init_logger, finalize_log


DEFAULT_DATA_PATH = 'MuSiQue/musique_sample_200.json'
DEFAULT_DB_PATH = 'MuSiQue/metadata_v4aligned.db'
DEFAULT_BM25_INDEX = 'MuSiQue/bm25_index_v4aligned'
DEFAULT_EMBEDDINGS = 'MuSiQue/path_embeddings_v4aligned.npz'

DEFAULT_OUTPUT_MUSIQUE = 'Results/test_musique_v12_ragpcot_results_v4aligned.json'
DEFAULT_OUTPUT_HOTPOT = 'Results/test_hotpot_v12_ragcot_results_v4aligned.json'

DATASET_DEFAULTS = {
    'musique': {
        'data_path': 'MuSiQue/musique_sample_200.json',
        'db_path': 'MuSiQue/metadata_v4aligned.db',
        'bm25_index': 'MuSiQue/bm25_index_v4aligned',
        'embeddings': 'MuSiQue/path_embeddings_v4aligned.npz',
    },
    'hotpot': {
        'data_path': 'HotpotQA/hotpotqa_sample_200.json',
        'db_path': 'HotpotQA/metadata_v4aligned.db',
        'bm25_index': 'HotpotQA/bm25_index_v4aligned',
        'embeddings': 'HotpotQA/path_embeddings_v4aligned.npz',
    },
}


def _json_default(obj):
    """Best-effort JSON serializer for numpy/scalars that may appear in retrieval results."""
    item = getattr(obj, 'item', None)
    if callable(item):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def parse_args():
    parser = argparse.ArgumentParser(description='Test v11 pipeline with rag_qa_cot one-shot (MuSiQue/HotpotQA)')
    parser.add_argument('--dataset', type=str, default='musique', choices=['musique', 'hotpot'])
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--db_path', type=str, default=DEFAULT_DB_PATH)
    parser.add_argument('--bm25_index', type=str, default=DEFAULT_BM25_INDEX)
    parser.add_argument('--embeddings', type=str, default=DEFAULT_EMBEDDINGS)
    parser.add_argument('--max_questions', type=int, default=None)
    parser.add_argument('--concurrency', type=int, default=100)
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_MUSIQUE} for musique, {DEFAULT_OUTPUT_HOTPOT} for hotpot)",
    )
    return parser.parse_args()


async def process_one(pipeline, item, idx, total):
    qid = item.get('_id', f'q{idx}')
    question = item['question']
    gold = item.get('answer')

    start = time.time()
    try:
        result = await pipeline.process_question(question)
        elapsed = time.time() - start

        if result.get('success'):
            pred = result.get('final_answer')
            num_passages = result.get('num_passages', 0)
            num_facts = result.get('num_facts', result.get('num_paths', None))
            facts_str = f", {num_facts}facts" if isinstance(num_facts, int) else ""
            print(f"[{idx+1:3d}/{total}] [OK] ({elapsed:.1f}s, {num_passages}p{facts_str}) {question[:60]}...")
            print(f"           Gold: {gold}")
            print(f"           Pred: {pred}")
            return {
                'id': qid,
                'question': question,
                'gold_answer': gold,
                'predicted_answer': pred,
                'answer_aliases': item.get('answer_aliases', []),
                'time': elapsed,
                'num_passages': num_passages,
                # Backward compatibility
                'num_paths': result.get('num_paths', None),
                # Preferred naming
                'num_facts': result.get('num_facts', result.get('num_paths', None)),
                'success': True,
                'decomposition': result.get('decomposition'),
            }

        print(f"[{idx+1:3d}/{total}] [FAIL] ({elapsed:.1f}s) {result.get('error')}")
        return {
            'id': qid,
            'question': question,
            'gold_answer': gold,
            'predicted_answer': None,
            'answer_aliases': item.get('answer_aliases', []),
            'time': elapsed,
            'success': False,
            'error': result.get('error'),
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"[{idx+1:3d}/{total}] [EXC] ({elapsed:.1f}s) {str(e)[:200]}")
        return {
            'id': qid,
            'question': question,
            'gold_answer': gold,
            'predicted_answer': None,
            'answer_aliases': item.get('answer_aliases', []),
            'time': elapsed,
            'success': False,
            'error': str(e),
        }


async def run_batches(pipeline, data, concurrency):
    results = []
    total = len(data)

    for start_idx in range(0, total, concurrency):
        batch = data[start_idx:start_idx + concurrency]
        tasks = [process_one(pipeline, item, start_idx + i, total) for i, item in enumerate(batch)]
        batch_results = await asyncio.gather(*tasks, return_exceptions=False)
        results.extend(batch_results)

    return results


async def main():
    args = parse_args()
    load_dotenv()

    if args.output is None:
        args.output = DEFAULT_OUTPUT_MUSIQUE if args.dataset == 'musique' else DEFAULT_OUTPUT_HOTPOT

    defaults = DATASET_DEFAULTS[args.dataset]
    if args.data_path == DEFAULT_DATA_PATH:
        args.data_path = defaults['data_path']
    if args.db_path == DEFAULT_DB_PATH:
        args.db_path = defaults['db_path']
    if args.bm25_index == DEFAULT_BM25_INDEX:
        args.bm25_index = defaults['bm25_index']
    if args.embeddings == DEFAULT_EMBEDDINGS:
        args.embeddings = defaults['embeddings']

    print('=' * 80)
    print(f"Dataset: {args.dataset} | v11 (Paths-as-Hints) + rag_qa_cot one-shot")
    print('=' * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {args.data_path}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max questions: {args.max_questions}")

    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL'),
    )

    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6,
        bm25_index_path=args.bm25_index,
        embeddings_path=args.embeddings,
    )

    pipeline = NewMultihopPipelineV11PathsHintCoT(
        client=client,
        retriever=retriever,
        data_path=args.data_path,
        db_path=args.db_path,
        top_k_passages=5,
        top_k_paths=30,
        path_fetch_k=50,
        verbose=False,
        log_messages=False,
        dataset_name=args.dataset,
    )

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.max_questions is not None:
        data = data[: max(0, args.max_questions)]

    Path('Results').mkdir(parents=True, exist_ok=True)

    init_logger(f"test_pipeline_paths_hint_cot_{args.dataset}")

    start = time.time()
    results = await run_batches(pipeline, data, args.concurrency)
    elapsed = time.time() - start

    pipeline.close()

    success_count = sum(1 for r in results if r.get('success'))
    fail_count = len(results) - success_count

    payload = {
        'meta': {
            'dataset': args.dataset,
            'pipeline': pipeline.__class__.__name__,
            'max_questions': args.max_questions,
            'concurrency': args.concurrency,
            'top_k_passages': 5,
            'top_k_paths': 30,
            'elapsed_sec': elapsed,
            'success': success_count,
            'fail': fail_count,
        },
        'results': results,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)

    finalize_log()

    print('-' * 80)
    print(f"Output: {args.output}")
    print(f"Elapsed: {elapsed:.1f}s | Success: {success_count} | Fail: {fail_count}")
    print('-' * 80)


if __name__ == '__main__':
    asyncio.run(main())
