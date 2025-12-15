#!/usr/bin/env python3
"""Run a single question through Pipeline v8 (MuSiQue by default)."""

import asyncio
import json
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from new_multihop_pipeline_v8 import NewMultihopPipelineV8
from new_multihop_pipeline_v7 import MetadataLinkerV7
from hybrid_path_retriever import HybridPathRetriever
from llm_logger import init_logger, finalize_log


async def main():
    load_dotenv()

    api_key = os.getenv("ALICE_OPENAI_KEY")
    chat_url = os.getenv("ALICE_CHAT_URL")
    if not api_key or not chat_url:
        raise RuntimeError("Missing ALICE_OPENAI_KEY or ALICE_CHAT_URL in environment")

    init_logger("Results/Logs")

    dataset_path = "MuSiQue/musique_sample_200.json"
    metadata_path = "MuSiQue/musique_sample_200_metadata.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    item = data[0]
    question = item["question"]
    print("Question:", question)

    client = AsyncOpenAI(api_key=api_key, base_url=chat_url)

    retriever = HybridPathRetriever(
        bm25_index_path="MuSiQue/bm25_index",
        embeddings_path="MuSiQue/path_embeddings.npz",
        dense_weight=0.6,
        bm25_weight=0.4,
    )

    linker = MetadataLinkerV7(metadata_path, max_docs_per_value=200)

    pipeline = NewMultihopPipelineV8(
        client=client,
        retriever=retriever,
        linker=linker,
        hotpotqa_path=dataset_path,
        top_k=3,
        verbose=True,
        top_k_fallback=10,
    )

    result = await pipeline.run(question)

    print("\nPredicted:", result.get("predicted_answer"))

    finalize_log()


if __name__ == "__main__":
    asyncio.run(main())
