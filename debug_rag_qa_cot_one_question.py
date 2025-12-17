#!/usr/bin/env python3
"""\
Debug: rag_qa_cot one-shot prompt wiring (single MuSiQue question)
=================================================================

Runs ONE question through `MuSiQueMultihopPipelineV11PathsHint` and logs the *actual*
chat messages sent to the LLM (including one-shot examples) for:
- sub-question answering calls
- final main-query answering call

Usage:
  python debug_rag_qa_cot_one_question.py
  python debug_rag_qa_cot_one_question.py --index 0

Env:
  ALICE_OPENAI_KEY, ALICE_CHAT_URL, ALICE_EMBED_URL
"""

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from hybrid_path_retriever import HybridPathRetriever
from llm_logger import init_logger, finalize_log
from new_multihop_pipeline_musique_paths_hint import MuSiQueMultihopPipelineV11PathsHint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, default=0, help='Question index in MuSiQue sample file')
    return p.parse_args()


async def main():
    args = parse_args()
    load_dotenv()

    init_logger()

    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL'),
    )

    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6,
        bm25_index_path='MuSiQue/bm25_index',
        embeddings_path='MuSiQue/path_embeddings.npz',
    )

    pipeline = MuSiQueMultihopPipelineV11PathsHint(
        client=client,
        retriever=retriever,
        musique_path='MuSiQue/musique_sample_200.json',
        db_path='MuSiQue/metadata_v3.db',
        verbose=True,
        log_messages=True,
    )

    with open('MuSiQue/musique_sample_200.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    item = data[args.index]
    print('\n' + '=' * 80)
    print(f"QID: {item.get('_id')}\nQ: {item.get('question')}\nGold: {item.get('answer')}")
    print('=' * 80 + '\n')

    res = await pipeline.process_question(item['question'])
    if res.get('success'):
        print(f"\n[FINAL ANSWER] {res.get('final_answer')}")
    else:
        print(f"\n[ERROR] {res.get('error')}")

    pipeline.close()
    log_path = finalize_log()
    print(f"\n[LOG] {log_path}")


if __name__ == '__main__':
    asyncio.run(main())
