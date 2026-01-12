import asyncio
import json
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


from openai import AsyncOpenAI

from new_multihop_pipeline_paths_hint_expansion import NewMultihopPipelineV11PathsHint


def main() -> None:
    data_path = repo_root / "LVEVAL" / "lveval_qa_compact.json"
    db_path = repo_root / "LVEVAL" / "metadata_v5.db"
    corpus_path = repo_root / "LVEVAL" / "lveval_corpus.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing QA file: {data_path}")
    if not corpus_path.exists():
        raise FileNotFoundError("Missing corpus file: LVEVAL/lveval_corpus.json")

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not corpus:
        raise RuntimeError(f"Empty corpus file: {corpus_path}")

    # Client/retriever are not needed for this check; we only validate that passage indices load.
    client = AsyncOpenAI(api_key="DUMMY", base_url="http://localhost")

    pipe = NewMultihopPipelineV11PathsHint(
        client=client,
        retriever=None,
        hotpotqa_path=str(data_path),
        db_path=str(db_path),
        verbose=False,
        top_k_passages=5,
        top_k_paths=30,
        path_fetch_k=50,
    )

    # Pick a few doc_ids from the corpus and verify we can fetch non-empty text.
    sample = corpus[:5]
    ok = 0
    for item in sample:
        doc_id = str(item.get("idx"))
        expected = (item.get("text") or "").strip()
        got = (pipe.get_original_passage_by_doc_id(doc_id) or "").strip()
        print("doc_id:", doc_id, "expected_len:", len(expected), "got_len:", len(got))
        if expected and got:
            ok += 1
    print("doc_id_passages_loaded:", len(getattr(pipe, "doc_id_passages", {}) or {}))
    print("non_empty_matches_in_sample:", ok, "/", len(sample))

    pipe.close()


if __name__ == "__main__":
    main()
