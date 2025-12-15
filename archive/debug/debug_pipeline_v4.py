#!/usr/bin/env python3
"""
Debug Pipeline V4
=================
Runs a few questions with verbose logging to debug "Insufficient information" issues.
"""

import asyncio
import json
import time
import os
import argparse
from dotenv import load_dotenv
from openai import AsyncOpenAI

from new_multihop_pipeline_v4 import NewMultihopPipelineV4, MetadataLinker, PassageReranker
from hybrid_path_retriever import HybridPathRetriever

# Dataset configurations
DATASET_CONFIGS = {
    'hotpotqa': {
        'data_path': 'HotpotQA/hotpotqa_sample_200.json',
        'metadata_json': 'HotpotQA/hotpotqa_sample_200_metadata.json',
        'passage_embeddings': 'HotpotQA/passage_embeddings.npz',
        'bm25_index': 'HotpotQA/bm25_index',
        'path_embeddings': 'HotpotQA/path_embeddings.npz',
    },
    'musique': {
        'data_path': 'MuSiQue/musique_sample_200.json',
        'metadata_json': 'MuSiQue/musique_sample_200_metadata.json',
        'passage_embeddings': 'MuSiQue/passage_embeddings.npz',
        'bm25_index': 'MuSiQue/bm25_index',
        'path_embeddings': 'MuSiQue/path_embeddings.npz',
    }
}

async def main():
    parser = argparse.ArgumentParser(description="Debug Pipeline V4")
    parser.add_argument("--dataset", type=str, default="musique", choices=["hotpotqa", "musique"])
    parser.add_argument("--limit", type=int, default=3, help="Number of questions to debug")
    args = parser.parse_args()
    
    load_dotenv()
    
    config = DATASET_CONFIGS[args.dataset]
    print(f"Debugging {args.dataset.upper()}...")
    
    # Initialize Clients
    chat_client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    embed_client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_EMBED_URL')
    )
    
    # Initialize Components
    print("Initializing Components...")
    retriever = HybridPathRetriever(
        bm25_index_path=config['bm25_index'],
        embeddings_path=config['path_embeddings']
    )
    
    linker = MetadataLinker(config['metadata_json'])
    
    if not os.path.exists(config['passage_embeddings']):
        print(f"Error: Passage embeddings not found at {config['passage_embeddings']}")
        return
        
    reranker = PassageReranker(config['passage_embeddings'], embed_client)
    
    # Initialize Pipeline with VERBOSE=True
    pipeline = NewMultihopPipelineV4(
        client=chat_client,
        retriever=retriever,
        linker=linker,
        reranker=reranker,
        hotpotqa_path=config['data_path'],
        top_k=3,
        verbose=True  # ENABLE VERBOSE LOGGING
    )
    
    # Load Data
    with open(config['data_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Run Debug Loop
    for i, item in enumerate(data[:args.limit]):
        print(f"\n{'='*80}")
        print(f"DEBUG QUESTION {i+1}: {item['question']}")
        print(f"{'='*80}")
        
        start = time.time()
        result = await pipeline.run(item['question'])
        elapsed = time.time() - start
        
        print(f"\nResult ({elapsed:.2f}s):")
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Gold: {item['answer']}")
        
        if result['predicted_answer'] == "Decomposition failed.":
            print(f"Error: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())
