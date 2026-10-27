# Paper-to-code audit

Audit: 2026-10-26. Reference: `C:/Development/Paper/TRA-RAG.pdf` (28 pages),
*Retrieval-Augmented Generation with Retrieval-optimized TRA Structure*.
Upstream base: `61039bfd650ebf1d950d265a1d6ce53173e19380`.
The original workspace contains older code and uncommitted experiments; it is preserved.

| Paper component | Required code |
|---|---|
| Section 3.1 / A.1: TRA construction | `build_metadata.py`, `setup_hotpotqa_db.py`, `Prompt/metadata_construction_prompt.py` |
| Section 3.2: Fact indexing | `embedding_text_generator.py`, `path_embedding_generator.py`, `bm25_indexer.py`, `setup_indices_v5.py` |
| Section 3.3 / A.2: QD | `query_decomposition.py`, `Prompt/query_decomposition_prompt.py` |
| Section 3.4: RRF and reranking | `hybrid_path_retriever.py`, `new_multihop_pipeline_paths_hint_expansion.py` |
| Multi-hop execution | `test_pipeline_paths_hint_expansion.py`, `Prompt/answer_prompt.py` |
| Simple QA | `run_simple_qa.py`, `prepare_singlehop_v5.py` |
| Metrics | `evaluate_mrqa.py`, `evaluate_retrieval.py`, `llm_evaluation.py` |
| Oracle / structural comparison | `upper_bound_experiment.py`, `compare_2wiki_triplets_vs_metadata.py`, `Prompt/compare_evaluation.py` |
| Appendix D sensitivity | `run_2wiki_reviewer_sensitivity.py` (requires cached experiment inputs) |
| Runtime support | `llm_provider.py`, `llm_logger.py` |
| Input preparation / ID maintenance | `prepare_lveval_v5.py`, `prepare_popqa_canonical_artifacts.py`, `singlehop_canonical.py` |
| Diagnostic | `evaluate_singlehop_answer_bearing_hit.py` (not evidence recall) |

The upstream standalone v12 already includes its v11 base. Copying the local v12
subclass alone would introduce an unnecessary dependency on the old v11 file.

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

- Section 3.2 describes one Fact per top-level attribute. Existing code splits lists
  of dictionaries into item-level Facts, while retaining nested dictionaries and
  grouped relation targets. Preserve this behavior until original indices are audited.
- Reranking uses SQ-max pooling per feature and candidate-set min-max normalization.
  The paper's per-SQ formula does not fully specify that pooling detail.
- The new Simple QA runner follows direct retrieval and reuses the final-answer prompt
  without an SQ chain. Its scores have not been checked against the original runs.
- The structural comparison utility defaults to 2Wiki; Table 4 describes HotpotQA.
  Its default execution does not reproduce Table 4 without original inputs and judge order.
- Exact 1,000-query subsets, 124 LV-Eval queries, generated TRA, embeddings, cached
  QD and model responses are not all versioned. Code alone cannot reproduce table values.
- Answer-bearing hits, canonical IDs and evidence recall are different protocols.

## Excluded material

The original workspace retains old pipelines, `archive/`, temporary case-search scripts,
downloaded third-party code, results, logs, databases, embeddings, credentials and virtual
environments. They are excluded from this curated release. Third-party baseline implementations
and full experiment outputs are not bundled: this is the core-method release, not a complete
reproduction package for every comparison row. Existing upstream MuSiQue inputs are retained.
The manuscript itself and additional local datasets are not uploaded.
