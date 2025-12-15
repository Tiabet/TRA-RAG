#!/usr/bin/env python3
"""Run a single question through Pipeline v9 (MuSiQue by default)."""

import asyncio
import json
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from hybrid_path_retriever import HybridPathRetriever
from new_pipeline_v9 import PipelineV9, MetadataLinkerFull
from llm_logger import init_logger, finalize_log


async def main():
    load_dotenv()
    api_key = os.getenv("ALICE_OPENAI_KEY")
    chat_url = os.getenv("ALICE_CHAT_URL")
    if not api_key or not chat_url:
        raise RuntimeError("Missing ALICE_OPENAI_KEY or ALICE_CHAT_URL")

    init_logger("Results/Logs")

    dataset_path = "MuSiQue/musique_sample_200.json"
    metadata_path = "MuSiQue/musique_sample_200_metadata.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    item = data[0]
    q = item["question"]
    print("Question:", q)

    client = AsyncOpenAI(api_key=api_key, base_url=chat_url)
    retriever = HybridPathRetriever(
        bm25_index_path="MuSiQue/bm25_index",
        embeddings_path="MuSiQue/path_embeddings.npz",
        dense_weight=0.6,
        bm25_weight=0.4,
    )

    linker = MetadataLinkerFull(metadata_path)
    pipeline = PipelineV9(
        client=client,
        retriever=retriever,
        linker=linker,
        dataset_path=dataset_path,
        initial_top_paths=3,
        final_top_passages=10,
        verbose=True,
    )

    out = await pipeline.run(q)
    print("\nPredicted:", out.get("predicted_answer"))
    print("Retrieved passages:", len(out.get("retrieved_passages", []) or []))
    ri = out.get("retrieval_info", {})
    print("Neighborhood titles:", ri.get("neighborhood_titles"))
    print("Candidate paths:", ri.get("candidate_paths"))

    finalize_log()


if __name__ == "__main__":
    asyncio.run(main())
