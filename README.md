# TRA-RAG

Code for **Retrieval-Augmented Generation with Retrieval-optimized TRA Structure**.
TRA organizes documents into Title, Relations, and Attributes. Serialized Facts are
retrieved with BM25 and dense embeddings. Multi-hop QA adds dependency-aware QD
and final reranking; Simple QA retrieves directly.

See [paper-to-code audit](docs/PAPER_CODE_MAP.md) for the retained files, manuscript
differences, and reproducibility limits. Historical names `metadata` and `path`
refer to TRA structures and Facts.

## Install

Python 3.10+:

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set credentials. Direct OpenAI and an
OpenAI-compatible ALICE gateway are supported. ALICE takes precedence if both are set.
Index construction, query embeddings, answering, and LLM judging may incur API costs.

## Data and indexing

Preserve document IDs across QA, corpus, metadata, and indices.

| Dataset | Indexing input | Question input | QA mode |
|---|---|---|---|
| HotpotQA | `HotpotQA/hotpotqa.json` | same, with context | multi-hop `hotpot` |
| MuSiQue | `MuSiQue/musique.json` | same, with paragraphs | multi-hop `musique` |
| 2Wiki | `2WikiMultihopQA/2wikimultihopqa.json` | QA list via `--data_path` | multi-hop `2wiki` |
| LV-Eval | `LVEVAL/lveval_corpus_for_pipeline.json` | `LVEVAL/lveval_qa_compact.json` | multi-hop `lveval` |
| NQ | `NQ_REAR/nq_rear_corpus_for_pipeline.json` | `NQ_REAR/nq_rear_qa.json` | simple `nq_rear` |
| PopQA | `POPQA/popqa_corpus_for_pipeline.json` | `POPQA/popqa_qa.json` | simple `popqa` |

The existing upstream MuSiQue inputs are retained. Other datasets and generated
artifacts must be supplied separately. QA records use `question`, `answer`, and
`id`/`_id`. Corpus layouts include HotpotQA `context` pairs and MuSiQue
`paragraphs` with `title`, `paragraph_text`, and stable `corpus_idx`.

`prepare_singlehop_v5.py` converts `NQ_REAR/nq_rear.json` + `nq_rear_corpus.json`
and `POPQA/popqa.json` + `popqa_corpus.json` into separate QA and corpus wrappers.
These inputs follow the local experiment's HippoRAG2-style format, not arbitrary raw downloads.
See `prepare_lveval_v5.py` for LV-Eval source filenames.

```bash
python prepare_singlehop_v5.py --dataset all
python setup_indices_v5.py --dataset musique --model gpt-4o-mini
```

For a gateway, supply its model ID, e.g. `--model openai/gpt-4o-mini`.
Indexing dataset names: `hotpotqa`, `musique`, `2wiki`, `lveval`, `nq_rear`, `popqa`.
Outputs per dataset: `metadata_v5.json`, `metadata_v5.db`,
`embedding_texts_v5.json`, `path_embeddings_v5.npz`, `bm25_index_v5/`.
Existing artifacts are reused; `--rebuild` overwrites them.

## Run

Multi-hop:

```bash
python test_pipeline_paths_hint_expansion.py --dataset musique --concurrency 5 --limit 10
```

Defaults: RRF k=60, sparse/dense weights 1.0/1.3, 300 candidates per retriever,
50 retained SQ Facts, final 3 seed + 2 reranked passages and 20 seed + 10 reranked Facts.
Override inputs with `--data_path`, `--db_path`, `--bm25_index_path`, `--embeddings_path`.

Simple QA (direct RRF, one final answer call, no QD or reranking):

```bash
python run_simple_qa.py --dataset nq_rear --limit 10
python run_simple_qa.py --dataset popqa --limit 10
```

`--no_llm` on the multi-hop runner and `--retrieval_only` on the simple runner skip
chat, but embeddings may still call an API. Multi-hop `--no_llm` also bypasses QD:
it is a retrieval smoke check, not full multi-hop evaluation. Its predictions are
null and must not be used to report QA accuracy.

## Evaluate

```bash
python evaluate_mrqa.py Results/musique_result.json
python evaluate_retrieval.py --result_path Results/musique_result.json --gold_path MuSiQue/musique.json --gold_id corpus_idx --at_k 5
python llm_evaluation.py --pred Results/musique_result.json --gold MuSiQue/musique.json
```

Choose the ID scheme matching the gold data. LLM evaluation calls an API.
`evaluate_singlehop_answer_bearing_hit.py` is a diagnostic, not evidence Recall@5.
LV-Eval lacks gold evidence in the paper and has no evidence Recall@5 or Oracle score.

## Ablations and validation

The multi-hop runner exposes `--sq_fusion_method`, `--final_selection_mode`,
`--no_previous_context`, `--top_k_paths`, and `--top_k_passages`.
`run_2wiki_reviewer_sensitivity.py` requires original cached QD results and artifacts.
`upper_bound_experiment.py` and `compare_2wiki_triplets_vs_metadata.py` retain
Oracle and structural-comparison utilities. Consult their `--help` first.

```bash
python -m unittest discover -s tests -v
```

Offline tests cover candidate budgets, direct Simple QA, null retrieval-only predictions,
and nested/grouped Fact preservation. Full benchmark scores have not been rerun.
Exact experiment replay needs original subsets, generated artifacts and model responses.
