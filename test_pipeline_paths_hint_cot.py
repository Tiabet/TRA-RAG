#!/usr/bin/env python3
"""\
Test New Multi-hop Pipeline v11 (Paths-as-Hints)
====================================================================
Unified pipeline test script for HotpotQA and MuSiQue datasets.

This mirrors `test_pipeline_no_link.py` behavior/configs as closely as possible,
but uses `new_multihop_pipeline_paths_hint_cot.py` for prompt-only experiments.

Usage:
        python test_pipeline_paths_hint_cot.py                     # Default: HotpotQA
        python test_pipeline_paths_hint_cot.py --dataset hotpotqa  # HotpotQA
        python test_pipeline_paths_hint_cot.py --dataset musique   # MuSiQue
        python test_pipeline_paths_hint_cot.py --dataset hotpotqa --artifacts v5 --max_questions 5
        python test_pipeline_paths_hint_cot.py --dataset musique --artifacts v5 --max_questions 5
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

from new_multihop_pipeline_paths_hint_cot import NewMultihopPipelineV11PathsHintCoT
from hybrid_path_retriever import HybridPathRetriever
from llm_logger import init_logger, finalize_log


# Concurrency settings
CONCURRENCY = 100

# Dataset configurations (same keys as test_pipeline_no_link.py)
DATASET_CONFIGS = {
    # Legacy artifacts (title-based / v4aligned)
    'v4aligned': {
        'hotpotqa': {
            'data_path': 'HotpotQA/hotpotqa_sample_200.json',
            'db_path': 'HotpotQA/metadata_v4aligned.db',
            'bm25_index': 'HotpotQA/bm25_index_v4aligned',
            'embeddings': 'HotpotQA/path_embeddings_v4aligned.npz',
            'result_path': 'Results/test_hotpot_v11_200_results_v4aligned_cot.json',
        },
        'musique': {
            'data_path': 'MuSiQue/musique_sample_200.json',
            'db_path': 'MuSiQue/metadata_v4aligned.db',
            'bm25_index': 'MuSiQue/bm25_index_v4aligned',
            'embeddings': 'MuSiQue/path_embeddings_v4aligned.npz',
            'result_path': 'Results/test_musique_v11_200_results_v4aligned_cot.json',
        },
    },
    # v5 artifacts (corpus_idx/doc_id unified)
    'v5': {
        'hotpotqa': {
            'data_path': 'HotpotQA/hotpotqa_sample_200_corpus_idx.json',
            'db_path': 'HotpotQA/metadata_v5.db',
            'bm25_index': 'HotpotQA/bm25_index_v5',
            'embeddings': 'HotpotQA/path_embeddings_v5.npz',
            'result_path': 'Results/test_hotpot_v11_200_results_v52_cot.json',
        },
        'musique': {
            'data_path': 'MuSiQue/musique_sample_200_corpus_idx.json',
            'db_path': 'MuSiQue/metadata_v5.db',
            'bm25_index': 'MuSiQue/bm25_index_v5',
            'embeddings': 'MuSiQue/path_embeddings_v5.npz',
            'result_path': 'Results/test_musique_v11_200_results_v52_cot.json',
        },
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
    parser = argparse.ArgumentParser(description='Test Pipeline v11 (paths-as-hints)')
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['hotpotqa', 'musique'])
    parser.add_argument('--result_path', type=str, default=None,
                        help='Optional override for output results JSON path')
    parser.add_argument('--artifacts', type=str, default='v5', choices=['v4aligned', 'v5'],
                        help='Which artifact set to use (default: v5)')
    parser.add_argument('--max_questions', type=int, default=None,
                        help='Optional limit for number of questions to run (useful for smoke tests)')
    parser.add_argument('--concurrency', type=int, default=CONCURRENCY,
                        help=f'Concurrency level (default: {CONCURRENCY})')
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
                'final_retrieved_passages': result.get('final_retrieved_passages', None),
                'final_retrieved_paths': result.get('final_retrieved_paths', None),
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

    dataset = args.dataset.lower()
    artifact_set = args.artifacts
    config = DATASET_CONFIGS[artifact_set][dataset]

    print('=' * 80)
    print(f"Multi-hop Pipeline V11 Test (CoT prompt) - {dataset.upper()}")
    print('=' * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset}")
    print(f"Artifacts: {artifact_set}")
    print(f"Data: {config['data_path']}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max questions: {args.max_questions}")

    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL'),
    )

    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6,
        bm25_index_path=config['bm25_index'],
        embeddings_path=config['embeddings'],
    )

    pipeline = NewMultihopPipelineV11PathsHintCoT(
        client=client,
        retriever=retriever,
        data_path=config['data_path'],
        db_path=config['db_path'],
        top_k_passages=5,
        top_k_paths=30,
        path_fetch_k=50,
        verbose=False,
        log_messages=False,
        dataset_name=dataset,
    )

    with open(config['data_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.max_questions is not None:
        data = data[: max(0, args.max_questions)]

    Path('Results').mkdir(parents=True, exist_ok=True)

    init_logger(f"test_pipeline_paths_hint_cot_{dataset}_{artifact_set}")

    start = time.time()
    results = await run_batches(pipeline, data, args.concurrency)
    elapsed = time.time() - start

    pipeline.close()

    success_count = sum(1 for r in results if r.get('success'))
    fail_count = len(results) - success_count

    payload = {
        'meta': {
            'dataset': dataset,
            'pipeline': pipeline.__class__.__name__,
            'max_questions': args.max_questions,
            'concurrency': args.concurrency,
            'top_k_passages': 5,
            'top_k_paths': 30,
            'path_fetch_k_input': 50,
            'path_fetch_k_effective': getattr(pipeline, 'path_fetch_k', None),
            'elapsed_sec': elapsed,
            'success': success_count,
            'fail': fail_count,
        },
        'results': results,
    }

    output_file = args.result_path or config['result_path']
    if args.max_questions is not None and args.result_path is None:
        stem, ext = os.path.splitext(output_file)
        output_file = f"{stem}_limit{int(args.max_questions)}{ext}"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)

    finalize_log()

    print('-' * 80)
    print(f"Output: {output_file}")
    print(f"Elapsed: {elapsed:.1f}s | Success: {success_count} | Fail: {fail_count}")
    print('-' * 80)


if __name__ == '__main__':
    asyncio.run(main())
