import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from new_multihop_pipeline_v3 import NewMultihopPipelineV3
from hybrid_path_retriever import HybridPathRetriever

async def main():
    load_dotenv()
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6,
        bm25_index_path='HotpotQA/bm25_index',
        embeddings_path='HotpotQA/path_embeddings.npz'
    )
    pipeline = NewMultihopPipelineV3(
        client=client,
        retriever=retriever,
        hotpotqa_path='HotpotQA/hotpotqa_sample_200.json',
        db_path='HotpotQA/metadata_v3.db',
        verbose=True
    )
    
    q = "Meet Market is a 2004 film starring which son of a former prime minister?"
    print(f"Processing: {q}")
    result = await pipeline.process_question(q)
    print("Result:", result['final_answer'])

if __name__ == "__main__":
    asyncio.run(main())
