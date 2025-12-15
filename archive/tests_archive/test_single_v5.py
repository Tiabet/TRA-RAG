#!/usr/bin/env python3
"""
Test Single Question on V5 Pipeline
"""

import asyncio
import json
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from new_multihop_pipeline_v5 import NewMultihopPipelineV5, MetadataLinkerV5
from hybrid_path_retriever import HybridPathRetriever
from llm_logger import init_logger, finalize_log

async def main():
    # Initialize Logger
    init_logger()
    
    load_dotenv()
    
    # Check API Key
    api_key = os.getenv('ALICE_OPENAI_KEY')
    chat_url = os.getenv('ALICE_CHAT_URL')
    
    if not api_key:
        print("Error: ALICE_OPENAI_KEY not found in .env")
        return

    # Initialize Chat Client
    chat_client = AsyncOpenAI(
        api_key=api_key,
        base_url=chat_url
    )
    
    # Config for HotpotQA
    hotpot_path = 'HotpotQA/hotpotqa_sample_200.json'
    metadata_path = 'HotpotQA/hotpotqa_sample_200_metadata.json'
    
    # Initialize components
    print("Initializing components...")
    retriever = HybridPathRetriever(
        bm25_index_path='HotpotQA/bm25_index',
        embeddings_path='HotpotQA/path_embeddings.npz'
    )
    
    linker = MetadataLinkerV5(metadata_path)
    
    pipeline = NewMultihopPipelineV5(
        client=chat_client,
        retriever=retriever,
        linker=linker,
        hotpotqa_path=hotpot_path,
        top_k=3,
        verbose=True
    )
    
    # Load one question
    with open(hotpot_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        item = data[0] # First question
        
    query = item['question']
    print(f"\nRunning pipeline for question: {query}")
    
    start_time = asyncio.get_event_loop().time()
    result = await pipeline.run(query)
    end_time = asyncio.get_event_loop().time()
    
    print(f"\nTotal time: {end_time - start_time:.2f}s")
    print(f"Predicted Answer: {result.get('predicted_answer')}")
    print(f"Ground Truth: {item.get('answer')}")
    
    # Finalize Log
    log_path = finalize_log()
    if log_path:
        print(f"Log saved to: {log_path}")

if __name__ == "__main__":
    asyncio.run(main())
