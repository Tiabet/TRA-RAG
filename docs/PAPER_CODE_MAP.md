# Paper-to-code audit

Updated: 2026-10-27. Authoritative reference: the 28-page
[EMNLP 2026 published paper](https://aclanthology.org/2026.emnlp-main.118/),
*Retrieval-Augmented Generation with Retrieval-optimized TRA Structure*, pp. 2735–2762.
Latest base: `b917afb` (merged first cleanup). The earlier local PDF is superseded
as the documentation reference. Sections 3–4 and Appendix B/D were checked against
the published PDF; the core procedure and main hyperparameters remain the same.
The original workspace contains older code and uncommitted experiments; it is preserved.

| Paper component | Required code |
|---|---|
| Section 3.1 / A.1: TRA construction | `build_metadata.py`, `metadata_database.py`, `Prompt/metadata_construction_prompt.py` |
| Section 3.2: Fact indexing | `fact_serialization.py`, `fact_embeddings.py`, `bm25_indexer.py`, `build_indices.py` |
| Section 3.3 / A.2: QD | `query_decomposition.py`, `Prompt/query_decomposition_prompt.py` |
| Section 3.4: RRF and reranking | `hybrid_retriever.py`, `tra_pipeline.py` |
| Multi-hop execution | `run_multihop_qa.py`, `multihop_runner.py`, `Prompt/answer_prompt.py` |
| Simple QA | `run_simple_qa.py`, `prepare_singlehop.py` |
| Metrics | `evaluate_mrqa.py`, `evaluate_retrieval.py`, `evaluate_answers_llm.py` |
| Oracle / structural comparison | `run_oracle.py`, `compare_2wiki_triplets_vs_metadata.py`, `Prompt/compare_evaluation.py` |
| Appendix D sensitivity | `run_2wiki_sensitivity.py` (requires cached experiment inputs) |
| Runtime support | `llm_provider.py`, `llm_logger.py` |
| Input preparation / ID maintenance | `prepare_lveval.py`, `prepare_popqa_canonical_artifacts.py`, `singlehop_canonical.py` |
| Diagnostic | `evaluate_singlehop_answer_bearing_hit.py` (not evidence recall) |

`tra_pipeline.py` contains both `BaseQAPipeline` and `TRAPipeline`; no separate
historical pipeline module is required.

## Release changes

- Restore missing requirements and allow them through `.gitignore`; add credential placeholders.
- Add NQ/PopQA indexing registrations and a direct Simple QA runner.
- Set sparse/dense candidates to 300 each, as Section 4.4 specifies. Previously
  the 50-Fact SQ pool indirectly requested 500 via `top_k_paths * 10`.
  This is a behavior change. For historical comparison the class accepts
  `retrieval_candidates=500`; this cleanup does not claim unchanged benchmark scores.
- Replace gold-answer predictions in retrieval-only output with nulls.
- Share provider selection for construction and embeddings; remove key-fragment logging.
- Correct evaluation CLI examples and clarify that retrieval-only can call embeddings APIs.

## Differences and limits

- Section 3.2 describes one Fact per top-level attribute. Fresh builds now preserve
  complete lists with `top_level`; `legacy_items` preserves the former item-level
  splitting. Index manifests reject incompatible reuse. This changes fresh index
  contents, so it is not a claim of numerical equivalence to old experiment artifacts.
- Section 4.4 main settings and Table 5 caption use different seed ratios. Explicit
  `main` and `table5` budget profiles preserve both instead of silently conflating them.
- Reranking uses SQ-max pooling per feature and candidate-set min-max normalization.
  The paper's per-SQ formula does not fully specify that pooling detail.
  The retained RRF implementation uses zero-based ranks (best rank is 0); the
  paper gives the rank formula without explicitly fixing the indexing convention.
- The new Simple QA runner follows direct retrieval and reuses the final-answer prompt
  without an SQ chain. Its scores have not been checked against the original runs.
  It also supports no-QD comparisons on multi-hop datasets (Appendix D.2).
- The structural comparison utility defaults to 2Wiki; Table 4 describes HotpotQA.
  Its default execution does not reproduce Table 4 without original inputs and judge order.
- Exact 1,000-query subsets, 124 LV-Eval queries, generated TRA, embeddings, cached
  QD and model responses are not all versioned. Code alone cannot reproduce table values.
- Answer-bearing hits, canonical IDs and evidence recall are different protocols.
  Appendix B.1 specifies repaired NQ and canonical/deduplicated PopQA; exact original
  inputs must be supplied. The release does not manufacture replacement gold labels.

## Excluded material

The original workspace retains old pipelines, `archive/`, temporary case-search scripts,
downloaded third-party code, results, logs, databases, embeddings, credentials and virtual
environments. They are excluded from this curated release. Third-party baseline implementations
and full experiment outputs are not bundled: this is the core-method release, not a complete
reproduction package for every comparison row. Existing upstream MuSiQue inputs are retained.
The manuscript itself and additional local datasets are not uploaded.
