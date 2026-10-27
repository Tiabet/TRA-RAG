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

## Included data: MuSiQue only

**This repository includes only MuSiQue data.** The paper evaluates six benchmarks,
but HotpotQA, 2WikiMultiHopQA, LV-Eval, NQ and PopQA data are **not included**.
Dataset names accepted by the scripts indicate code support, not bundled data.

| Included file | Contents | Use |
|---|---|---|
| `MuSiQue/musique.json` | 1,000 QA records with questions, answers and paragraphs | Default input for indexing, QA and evaluation |
| `MuSiQue/musique_corpus.json` | 11,656 title/text records | Separate corpus file; not the default indexing or QA input |

The counts above describe the committed files, not every experiment corpus in the
paper. Preserve document IDs across QA, corpus, metadata and indices.
Generated TRA metadata, SQLite databases, embeddings and BM25 indices are not bundled;
build them before running QA. The following workflow uses the included MuSiQue input.

## Build MuSiQue indices

```bash
python setup_indices_v5.py --dataset musique
```

The model defaults to the configured provider's model; `--model` overrides it.
Outputs under `MuSiQue/`: `metadata_v5.json`, `metadata_v5.db`,
`embedding_texts_v5.json`, `path_embeddings_v5.npz`, `bm25_index_v5/`.
Existing artifacts are reused; `--rebuild` overwrites them.

New indices default to `--fact_granularity top_level`, matching Section 3.2:
one Fact contains a title, a top-level key, and its complete value, including lists.
Historical indices split lists of objects into separate items. Use
`--fact_granularity legacy_items` to keep that behavior. An `index_manifest.json`
records the policy. Unmanifested existing indices are treated as legacy;
switching policies requires backing up the artifacts and rebuilding both indices.

## Run MuSiQue

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

Optional no-QD comparison on the included MuSiQue input (direct RRF and one final answer call):

```bash
python run_simple_qa.py --dataset musique --limit 10
```

This mode omits sequential answering and reranking as described in Appendix D.2.

`--no_llm` on the multi-hop runner and `--retrieval_only` on the simple runner skip
chat, but embeddings may still call an API. Multi-hop `--no_llm` also bypasses QD:
it is a retrieval smoke check, not full multi-hop evaluation. Its predictions are
null and must not be used to report QA accuracy.

## Evaluate MuSiQue

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
`--budget_profile main` keeps 3+2 passages and 20+10 Facts. The code also retains
`--budget_profile table5`, but reproducing that table requires external 2Wiki data.

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

## Optional external datasets (not included)

The code retains adapters for HotpotQA, 2WikiMultiHopQA, LV-Eval, NQ and PopQA.
Their data must be obtained and prepared separately; selecting a dataset name does
not download it. Do not run `--dataset all` on a fresh clone containing only MuSiQue.

`prepare_singlehop_v5.py` expects pre-existing NQ/PopQA inputs in the local
experiment's HippoRAG2-style format. `prepare_lveval_v5.py` likewise expects external
LV-Eval files. Neither is a required step in the MuSiQue workflow above.
The published experiments use repaired NQ and canonical/deduplicated PopQA;
adapters alone do not supply those exact experiment inputs.

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
