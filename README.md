## TRA-RAG

This repository contains:

- A v5 indexing pipeline that builds metadata and retrieval artifacts.
- A v12 multi-hop QA pipeline (paths-as-hints) with final reranking.

Main workflows:

1) Build indices: `setup_indices_v5.py`
2) Run the pipeline: `test_pipeline_paths_hint_expansion.py`
3) Evaluate outputs: `evaluate_mrqa.py`, `evaluate_retrieval.py`, `llm_evaluation.py`

---

## Requirements

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment variables

This repo uses an OpenAI-compatible API client.

Create a `.env` file (or export environment variables) with:

### Metadata (chat) endpoint

- `ALICE_CHAT_URL` (example: `https://api.openai.com/v1`)
- `ALICE_OPENAI_KEY` (API key)

### Embeddings endpoint

- `ALICE_EMBED_URL` (example: `https://api.openai.com/v1`)
- `ALICE_OPENAI_KEY` (same key)

---

## Build v5 artifacts (indexing)

The main entry point is `setup_indices_v5.py`.

It builds, per dataset:

- `metadata_v5.json` (LLM-produced metadata)
- `metadata_v5.db` (SQLite DB)
- `embedding_texts_v5.json` (embedding-ready texts)
- `path_embeddings_v5.npz` (dense embeddings)
- `bm25_index_v5/` (BM25 index directory)

This workspace is currently set up to run on the **MuSiQue** dataset.

MuSiQue inputs:

- `MuSiQue/musique.json`

MuSiQue outputs (v5 artifacts):

- `MuSiQue/metadata_v5.json`
- `MuSiQue/metadata_v5.db`
- `MuSiQue/embedding_texts_v5.json`
- `MuSiQue/path_embeddings_v5.npz`
- `MuSiQue/bm25_index_v5/`

Build MuSiQue artifacts:

```bash
python setup_indices_v5.py --dataset musique
```

Rebuild (overwrite existing outputs):

```bash
python setup_indices_v5.py --dataset musique --rebuild
```

Dry-run metadata (no LLM calls):

```bash
python setup_indices_v5.py --dataset musique --dry_run_metadata
```

---

## Run the v12 QA pipeline

The runner is `test_pipeline_paths_hint_expansion.py`.

MuSiQue retrieval-only smoke test (no LLM calls):

```bash
python test_pipeline_paths_hint_expansion.py --dataset musique --no_llm --limit 20
```

Retrieval-only (no LLM calls):

```bash
python test_pipeline_paths_hint_expansion.py --dataset musique --no_llm
```

Full pipeline (uses LLM):

```bash
python test_pipeline_paths_hint_expansion.py --dataset musique --concurrency 20
```

Outputs are written to:

- `Results/musique_result.json`

Override artifact paths:

```bash
python test_pipeline_paths_hint_expansion.py \
  --dataset musique \
  --data_path MuSiQue/musique.json \
  --db_path MuSiQue/metadata_v5.db \
  --bm25_index_path MuSiQue/bm25_index_v5 \
  --embeddings_path MuSiQue/path_embeddings_v5.npz
```

---

## Evaluation

MRQA-style EM / F1:

```bash
python evaluate_mrqa.py Results/musique_result.json
```

Retrieval evaluation:

```bash
python evaluate_retrieval.py Results/musique_result.json
```

LLM-based evaluation:

```bash
python llm_evaluation.py Results/musique_result.json
```
