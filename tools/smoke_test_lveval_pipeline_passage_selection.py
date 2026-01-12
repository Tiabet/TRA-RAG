import asyncio
import json
import sys
from pathlib import Path

from openai import AsyncOpenAI

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from new_multihop_pipeline_paths_hint_expansion import NewMultihopPipelineV12PathsHintExpansion


class DummyRetriever:
    async def search_hybrid(
        self,
        query: str,
        top_k: int,
        bm25_candidates: int,
        dense_candidates: int,
        fusion_method: str = "rrf",
    ):
        # Return deterministic fake paths with real LVEVAL doc_ids.
        # The passage selection logic should now be able to resolve these doc_ids
        # into original_passage via the LVEVAL corpus fallback loader.
        k = min(int(top_k), 120)
        out = []
        for i in range(k):
            out.append(
                {
                    "index": i,
                    "doc_id": str(i),
                    "title": f"dummy_title_{i}",
                    "source_title": f"dummy_source_{i}",
                    "entity_title": f"dummy_entity_{i}",
                    "key_path": "dummy.key",
                    "value": "dummy_value",
                    "score": float(k - i),
                    "bm25_score": float(k - i),
                    "dense_score": 0.0,
                    "origin": "dummy",
                }
            )
        return out


def main() -> None:
    data_path = repo_root / "LVEVAL" / "lveval_qa_compact.json"
    db_path = repo_root / "LVEVAL" / "metadata_v5.db"
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    qa = json.loads(data_path.read_text(encoding="utf-8"))
    question = qa[0].get("question") or "dummy"

    client = AsyncOpenAI(api_key="DUMMY", base_url="http://localhost")

    pipeline = NewMultihopPipelineV12PathsHintExpansion(
        client=client,
        retriever=DummyRetriever(),
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
