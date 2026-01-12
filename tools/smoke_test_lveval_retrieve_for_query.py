import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from hybrid_path_retriever import HybridPathRetriever
from new_multihop_pipeline_paths_hint_expansion import NewMultihopPipelineV12PathsHintExpansion


def main() -> None:
    load_dotenv()

    data_path = repo_root / "LVEVAL" / "lveval_qa_compact.json"
    db_path = repo_root / "LVEVAL" / "metadata_v5.db"
    bm25_index_path = repo_root / "LVEVAL" / "bm25_index_v5"
    embeddings_path = repo_root / "LVEVAL" / "path_embeddings_v5.npz"

    qa = json.loads(data_path.read_text(encoding="utf-8"))
    question = qa[0]["question"]

    # LLM is not used in retrieve_for_query; pass dummy client.
    client = AsyncOpenAI(api_key=os.getenv("ALICE_OPENAI_KEY") or "DUMMY", base_url=os.getenv("ALICE_CHAT_URL") or "http://localhost")

    retriever = HybridPathRetriever(
        bm25_index_path=str(bm25_index_path),
        embeddings_path=str(embeddings_path),
        bm25_weight=1.0,
        dense_weight=1.3,
    )

    pipeline = NewMultihopPipelineV12PathsHintExpansion(
        client=client,
        retriever=retriever,
        hotpotqa_path=str(data_path),
        db_path=str(db_path),
        top_k_passages=5,
        top_k_paths=30,
        path_fetch_k=50,
        verbose=False,
    )

    passages, paths = asyncio.run(pipeline.retrieve_for_query(question))

    print("question:", question)
    print("num_paths:", len(paths or []))
    print("num_passages:", len(passages or []))
    if passages:
        p0 = passages[0]
        print("first_passage_doc_id:", p0.get("doc_id"))
        print("first_passage_title:", p0.get("title"))
        text = (p0.get("original_passage") or "")
        print("first_passage_text_len:", len(text))
        print("first_passage_text_prefix:", text[:160].replace("\n", " "))

    pipeline.close()


if __name__ == "__main__":
    main()
