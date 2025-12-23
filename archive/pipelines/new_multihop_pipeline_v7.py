#!/usr/bin/env python3
"""
New Multi-hop Pipeline v7
==========================

Delta vs v6
-----------
Expansion noise filters (only 1+2, key-value linking kept for later):
1) Exclude values coming from relations.relation (relation label) from expansion linking.
2) Degree cutoff: block values that connect to too many documents (high-degree hubs).

Retrieval Strategy (per sub-question)
-----------------------------------
1) Initial Retrieval: Top-k unique docs by SQ (effective_query)
2) Metadata Expansion: find linked docs from initial docs, but filtered
3) Expanded Reranking: score ONLY expanded candidates by main_query
4) Final Passages: initial_top_k + expanded_top_k (unique)
"""

import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from openai import AsyncOpenAI

from query_decomposition import (
    decompose_query,
    QueryDecomposition,
    SubQuestion,
    substitute_answers,
    get_execution_order,
)
from hybrid_path_retriever import HybridPathRetriever
from llm_logger import log_llm_call, log_llm_error

from Prompt.answer_prompt import (
    DETAILED_SUBQUESTION_ANSWERING_PROMPT,
    FINAL_ANSWER_SYNTHESIS_PROMPT,
)


class MetadataLinkerV7:
    """Metadata linker with expansion noise filters."""

    def __init__(self, metadata_path: str, max_docs_per_value: int = 200):
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.max_docs_per_value = max_docs_per_value

        # value -> titles, title -> values
        self.value_to_titles = defaultdict(set)
        self.title_to_values = defaultdict(set)

        # Blocked high-degree values (computed after build)
        self.blocked_values = set()

        self._build_graph()
        self._compute_blocked_values()

        print(
            f"✓ Metadata Graph built: {len(self.title_to_values)} documents, "
            f"{len(self.value_to_titles)} values (blocked: {len(self.blocked_values)})"
        )

    def normalize_text(self, text):
        if text is None:
            return ""
        s = str(text).lower()
        s = s.replace(",", "")
        return s.strip()

    def _extract_values(self, obj, values_set):
        if isinstance(obj, dict):
            for _, v in obj.items():
                self._extract_values(v, values_set)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_values(item, values_set)
        elif isinstance(obj, (str, int, float, bool)):
            val = self.normalize_text(obj)
            if val:
                values_set.add(val)

    def _build_graph(self):
        for item in self.metadata:
            context_metadata = item.get("context_metadata", [])
            for doc in context_metadata:
                meta = doc.get("metadata", {})
                if not meta:
                    continue

                title = meta.get("title") or doc.get("title")
                if not title:
                    continue

                values = set()
                values.add(self.normalize_text(title))

                # attributes: keep
                self._extract_values(meta.get("attributes", {}), values)

                # relations: keep target values only; explicitly ignore relation labels
                for rel in meta.get("relations", []):
                    if not isinstance(rel, dict):
                        continue

                    # v7 rule (1): do NOT add rel.get('relation') values
                    target = rel.get("target")
                    if target:
                        values.add(self.normalize_text(target))

                for val in values:
                    if not val:
                        continue
                    self.value_to_titles[val].add(title)
                    self.title_to_values[title].add(val)

    def _compute_blocked_values(self):
        if self.max_docs_per_value is None:
            return
        for val, titles in self.value_to_titles.items():
            if len(titles) > self.max_docs_per_value:
                self.blocked_values.add(val)

    def get_linked_info(self, title: str) -> List[Tuple[str, str]]:
        """Get (linked_title, shared_value) pairs, skipping blocked hub values."""
        linked_info: List[Tuple[str, str]] = []
        values = self.title_to_values.get(title, set())

        for val in values:
            if val in self.blocked_values:
                continue
            shared_docs = self.value_to_titles.get(val, set())
            for doc in shared_docs:
                if doc != title:
                    linked_info.append((doc, val))

        return linked_info


class NewMultihopPipelineV7:
    def __init__(
        self,
        client: AsyncOpenAI,
        retriever: HybridPathRetriever,
        linker: MetadataLinkerV7,
        hotpotqa_path: str,
        top_k: int = 3,
        verbose: bool = False,
    ):
        self.client = client
        self.retriever = retriever
        self.linker = linker
        self.top_k = top_k
        self.verbose = verbose

        self.original_passages = self._load_original_passages(hotpotqa_path)
        if self.verbose:
            print(f"✓ Loaded {len(self.original_passages)} original passages")

    def _load_original_passages(self, hotpotqa_path: str) -> Dict[str, str]:
        with open(hotpotqa_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        passages: Dict[str, str] = {}
        for item in data:
            for title, sentences in item.get("context", []):
                if title not in passages:
                    passages[title] = "".join(sentences).strip()
        return passages

    def get_original_passage(self, title: str) -> Optional[str]:
        return self.original_passages.get(title)

    async def retrieve_for_query(self, sq_query: str, main_query: str) -> Tuple[List[Dict], Dict]:
        """Initial retrieval by SQ; expansion reranking by main_query."""

        # 1) Initial Retrieval by SQ
        initial_paths = await self.retriever.search_hybrid(sq_query, top_k=self.top_k * 5)

        initial_passages: List[Dict] = []
        initial_titles = set()

        for p in initial_paths:
            title = p["title"]
            if title in initial_titles:
                continue

            initial_titles.add(title)
            original_passage = self.get_original_passage(title)

            initial_passages.append(
                {
                    "title": title,
                    "original_passage": original_passage,
                    "score": p["score"],
                    "matched_path": p["key_path"],
                    "matched_value": p["value"],
                    "dense_score": p["dense_score"],
                    "bm25_score": p["bm25_score"],
                    "dense_rank": p.get("dense_rank"),
                    "bm25_rank": p.get("bm25_rank"),
                    "source": "initial",
                }
            )

            if len(initial_passages) >= self.top_k:
                break

        # 2) Expansion with filters
        candidate_indices = set()
        expanded_titles_seen = set()

        blocked_value_hits = 0
        total_values_seen = 0

        for title in initial_titles:
            # Count how many values we skip due to blocking
            values = self.linker.title_to_values.get(title, set())
            for v in values:
                total_values_seen += 1
                if v in self.linker.blocked_values:
                    blocked_value_hits += 1

            linked_info = self.linker.get_linked_info(title)

            for linked_title, shared_value in linked_info:
                if linked_title in initial_titles:
                    continue

                expanded_titles_seen.add(linked_title)

                doc_indices = self.retriever.get_indices_for_title(linked_title)
                for idx in doc_indices:
                    path_val_raw = str(self.retriever.values[idx])
                    path_val_norm = self.linker.normalize_text(path_val_raw)
                    if shared_value in path_val_norm:
                        candidate_indices.add(idx)

        # 3) Expanded reranking by MAIN QUERY
        expanded_passages: List[Dict] = []
        if candidate_indices:
            scored_expanded_paths = await self.retriever.score_candidates_rrf(
                main_query,
                list(candidate_indices),
                top_k=self.top_k * 5,
            )

            seen_expanded_titles = set()
            for p in scored_expanded_paths:
                title = p["title"]
                if title in initial_titles or title in seen_expanded_titles:
                    continue

                seen_expanded_titles.add(title)
                original_passage = self.get_original_passage(title)

                expanded_passages.append(
                    {
                        "title": title,
                        "original_passage": original_passage,
                        "score": p["score"],
                        "matched_path": p["key_path"],
                        "matched_value": p["value"],
                        "dense_score": p["dense_score"],
                        "bm25_score": p["bm25_score"],
                        "dense_rank": p.get("dense_rank"),
                        "bm25_rank": p.get("bm25_rank"),
                        "source": "expanded",
                    }
                )

                if len(expanded_passages) >= self.top_k:
                    break

        final_passages = initial_passages + expanded_passages

        stats = {
            "initial_count": len(initial_passages),
            "expanded_candidates_docs": len(expanded_titles_seen),
            "expanded_candidates_paths": len(candidate_indices),
            "expanded_selected": len(expanded_passages),
            "total_final": len(final_passages),
            "initial_query": sq_query,
            "expanded_rerank_query": main_query,
            "blocked_values_total": len(self.linker.blocked_values),
            "blocked_value_hits": blocked_value_hits,
            "total_values_seen": total_values_seen,
            "max_docs_per_value": self.linker.max_docs_per_value,
        }

        return final_passages, stats

    def _build_simple_previous_context(self, current_sq: SubQuestion, decomposition: QueryDecomposition) -> str:
        if not current_sq.depends_on:
            return ""
        context_parts = []
        for dep_id in current_sq.depends_on:
            dep_sq = decomposition.get_subquestion(dep_id)
            if dep_sq and dep_sq.answer:
                context_parts.append(f"{dep_id}: {dep_sq.question}")
                context_parts.append(f"Answer: {dep_sq.answer}")
                context_parts.append("")
        if context_parts:
            return "Previous Sub-Questions:\n" + "\n".join(context_parts)
        return ""

    async def answer_subquestion(self, sq: SubQuestion, decomposition: QueryDecomposition) -> None:
        context_for_query = self._build_simple_previous_context(sq, decomposition)
        effective_query = substitute_answers(sq.question, decomposition.subquestions)

        if self.verbose:
            print(f"\n[SQ: {sq.id}] {effective_query}")

        passages, stats = await self.retrieve_for_query(effective_query, decomposition.main_query)
        sq.retrieved_passages = passages
        sq.retrieval_info = stats

        if not passages:
            sq.answer = "Insufficient information."
            return

        context_text = ""
        for i, p in enumerate(passages):
            context_text += f"Document {i+1} ({p['title']}):\n{p['original_passage']}\n\n"

        prompt = (
            DETAILED_SUBQUESTION_ANSWERING_PROMPT.replace("{{main_query}}", decomposition.main_query)
            .replace("{{subquestion}}", effective_query)
            .replace("{{passages}}", context_text)
            .replace("{{previous_context}}", context_for_query)
        )

        try:
            response = await self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise question answering system. Give short, direct answers.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip()
            sq.answer = answer

            log_llm_call(
                call_type=f"SubQuestion Answering ({sq.id})",
                input_text="OMITTED",
                output_text=answer,
                context={
                    "subquestion": effective_query,
                    "passages": context_text,
                    "previous_context": context_for_query,
                    "main_query": decomposition.main_query,
                    "retrieval_stats": stats,
                },
            )

            if self.verbose:
                print(f"  -> Answer: {answer}")

        except Exception as e:
            log_llm_error(
                call_type=f"SubQuestion Answering ({sq.id})",
                error=str(e),
                context={"subquestion": effective_query},
            )
            sq.answer = "Error generating answer."

    async def run(self, query: str) -> Dict:
        if self.verbose:
            print(f"Processing Query: {query}")

        decomposition_result = await decompose_query(self.client, query)
        if not decomposition_result or not decomposition_result.get("success"):
            return {"error": "Decomposition failed", "predicted_answer": "Decomposition failed."}

        decomposition = decomposition_result["decomposition"]

        if self.verbose:
            print("Decomposition:")
            for sq in decomposition.subquestions:
                print(f"  - {sq.id}: {sq.question}")

        execution_batches = get_execution_order(decomposition)
        for batch in execution_batches:
            for sq_id in batch:
                sq = decomposition.get_subquestion(sq_id)
                if sq:
                    await self.answer_subquestion(sq, decomposition)

        final_context = []
        all_passages = []
        seen_titles = set()

        for sq in decomposition.subquestions:
            if sq.answer:
                final_context.append(f"Q: {sq.question}\nA: {sq.answer}")
            if sq.retrieved_passages:
                for p in sq.retrieved_passages:
                    if p["title"] not in seen_titles:
                        all_passages.append(p)
                        seen_titles.add(p["title"])

        context_text = "\n\n".join(
            [f"Document ({p['title']}):\n{p['original_passage']}" for p in all_passages]
        )
        qa_history = "\n\n".join(final_context)

        final_prompt = (
            FINAL_ANSWER_SYNTHESIS_PROMPT.replace("{{main_question}}", query)
            .replace("{{subquestion_chain}}", qa_history)
            .replace("{{passages}}", context_text)
        )

        try:
            response = await self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise question answering system. Give short, direct answers.",
                    },
                    {"role": "user", "content": final_prompt},
                ],
                temperature=0.0,
            )
            final_answer = response.choices[0].message.content.strip()

            log_llm_call(
                call_type="Final Synthesis",
                input_text="OMITTED",
                output_text=final_answer,
                context={
                    "main_question": query,
                    "subquestion_chain": qa_history,
                    "passages": context_text,
                },
            )

        except Exception as e:
            log_llm_error(call_type="Final Synthesis", error=str(e), context={"main_question": query})
            final_answer = "Error generating final answer."

        return {
            "decomposition": decomposition.to_dict(),
            "predicted_answer": final_answer,
            "passages": all_passages,
        }
