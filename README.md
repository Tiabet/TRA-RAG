# TRA-RAG

Code for **Retrieval-Augmented Generation with Retrieval-optimized TRA Structure**.
Published at **EMNLP 2026 (Main Conference)**, pages 2735–2762.

[Paper](https://aclanthology.org/2026.emnlp-main.118/) ·
[PDF](https://aclanthology.org/2026.emnlp-main.118.pdf) ·
[Citation](CITATION.cff) · [Paper/code mapping](docs/PAPER_CODE_MAP.md)

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
python setup_indices_v5.py --dataset musique
```

The model defaults to the configured provider's model; `--model` overrides it.
Indexing dataset names: `hotpotqa`, `musique`, `2wiki`, `lveval`, `nq_rear`, `popqa`.
Outputs per dataset: `metadata_v5.json`, `metadata_v5.db`,
`embedding_texts_v5.json`, `path_embeddings_v5.npz`, `bm25_index_v5/`.
Existing artifacts are reused; `--rebuild` overwrites them.

New indices default to `--fact_granularity top_level`, matching Section 3.2:
one Fact contains a title, a top-level key, and its complete value, including lists.
Historical indices split lists of objects into separate items. Use
`--fact_granularity legacy_items` to keep that behavior. An `index_manifest.json`
records the policy. Unmanifested existing indices are treated as legacy;
switching policies requires backing up the artifacts and rebuilding both indices.

## Run

Multi-hop:

```bash
python run_multihop_qa.py --dataset musique --concurrency 5 --limit 10
```

Defaults: RRF k=60, sparse/dense weights 1.0/1.3, 300 candidates per retriever,
50 retained SQ Facts, final 3 seed + 2 reranked passages and 20 seed + 10 reranked Facts.
Override inputs with `--data_path`, `--corpus_path`, `--db_path`,
`--bm25_index_path`, `--embeddings_path`. Use `--corpus_path` when questions and
corpus records are separate. The original experiment runner
`test_pipeline_paths_hint_expansion.py` remains supported.

Simple QA (direct RRF, one final answer call, no QD or reranking):

```bash
python run_simple_qa.py --dataset nq_rear --limit 10
python run_simple_qa.py --dataset popqa --limit 10
```

For the paper's no-QD comparison on multi-hop data, the same direct runner supports
`hotpot`, `musique`, `2wiki`, and `lveval`:

```bash
python run_simple_qa.py --dataset 2wiki --qa_path 2WikiMultihopQA/2wiki_qa.json
```

This mode omits sequential answering and reranking as described in Appendix D.2.
NQ uses repaired input and PopQA uses canonical/deduplicated input in the published
experiments. Supply those exact inputs when comparing reported results; preparation
alone does not guarantee the same corpus or subset as the paper.

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
Section 4.4 and Table 5 have different seed allocations. The default
`--budget_profile main` keeps 3+2 passages and 20+10 Facts. For the Table 5 grid:

```bash
python run_multihop_qa.py --dataset 2wiki --budget_profile table5 --top_k_passages 5 --top_k_paths 30 --data_path 2WikiMultihopQA/2wiki_qa.json --corpus_path 2WikiMultihopQA/2wikimultihopqa.json
```

| Profile | Total passages → seed count | Total Facts → seed count |
|---|---|---|
| Main | 5 → 3 | 30 → 20 |
| Table 5 | 0/3/5/10 → 0/1/2/4 | 0/10/30/50 → 0/4/10/20 |

Remaining slots use reranking. `--retrieval_candidates` (default 300) and `--rrf_k`
(default 60) are explicit and recorded in multi-hop result configurations.
`run_2wiki_reviewer_sensitivity.py` requires original cached QD results and artifacts.
`upper_bound_experiment.py` and `compare_2wiki_triplets_vs_metadata.py` retain
Oracle and structural-comparison utilities. Consult their `--help` first.

```bash
python -m unittest discover -s tests -v
```

Offline tests cover candidate budgets, QD dependency order, direct Simple QA,
null retrieval-only predictions, Fact preservation, index compatibility and budget profiles.
Full benchmark scores have not been rerun.
Exact experiment replay needs original subsets, generated artifacts and model responses.

## Citation

```bibtex
@inproceedings{kim-etal-2026-retrieval-augmented,
  title = {Retrieval-Augmented Generation with Retrieval-optimized {TRA} Structure},
  author = {Kim, JungHyoun and Kim, Soohyeong and Hwang, Seok Jun and Park, Jeonghyeon and Choi, Yong Suk},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year = {2026},
  month = oct,
  address = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  pages = {2735--2762},
  url = {https://aclanthology.org/2026.emnlp-main.118/}
}
```
