#!/usr/bin/env python3
"""New Multi-hop Pipeline v8

Goal
----
Change the search scope over time using dependency structure:

- Independent SQs (depends_on == []): answer using global hybrid retrieval top-k titles.
- Dependent SQs: answer using ONLY the expansion pool created from its dependencies.
  (We score candidate *paths* within that pool via RRF and keep top-k unique titles.)

Pool update policy
------------------
B) After answering each SQ, expand from the top passages used for that SQ and
   create/refresh an expansion pool for that SQ.

Dependent pool source
---------------------
B) For a SQ with depends_on=[SQ1,SQ2], use the union of pools created by SQ1 and SQ2.

Fallbacks
---------
- If an independent SQ answer is Insufficient information:
  retry once with broader global retrieval (top_k_fallback).
  If still insufficient, its pool is left empty.

- If a dependent SQ pool is empty (or yields no passages):
  fall back to global retrieval top-k titles for that SQ.
"""

import json
from typing import Any, Dict, List, Optional, Set, Tuple

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

from new_multihop_pipeline_v7 import MetadataLinkerV7


def _is_insufficient(answer: Optional[str]) -> bool:
    if not answer:
        return True
    s = answer.strip().lower()
    return s.startswith("insufficient information") or s in {"insufficient", "unknown"}


class NewMultihopPipelineV8:
    def __init__(
        self,
        client: AsyncOpenAI,
        retriever: HybridPathRetriever,
        linker: MetadataLinkerV7,
        hotpotqa_path: str,
        top_k: int = 3,
        verbose: bool = False,
        top_k_fallback: int = 10,
    ):
        self.client = client
        self.retriever = retriever
        self.linker = linker
        self.top_k = top_k
        self.verbose = verbose
        self.top_k_fallback = top_k_fallback

        self.original_passages = self._load_original_passages(hotpotqa_path)

        # Per-SQ expansion pools: sq_id -> set(path_index)
        self.sq_pools: Dict[str, Set[int]] = {}

    def _load_original_passages(self, data_path: str) -> Dict[str, str]:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        passages: Dict[str, str] = {}
        for item in data:
            for title, sentences in item.get("context", []):
                if title not in passages:
                    passages[title] = "".join(sentences).strip()
        return passages

    def get_original_passage(self, title: str) -> Optional[str]:
        return self.original_passages.get(title)

    def _build_previous_context(self, current_sq: SubQuestion, decomposition: QueryDecomposition) -> str:
        if not current_sq.depends_on:
            return ""
        parts: List[str] = []
        for dep_id in current_sq.depends_on:
            dep_sq = decomposition.get_subquestion(dep_id)
            if dep_sq and dep_sq.answer:
                parts.append(f"{dep_id}: {dep_sq.question}")
                parts.append(f"Answer: {dep_sq.answer}")
                parts.append("")
        return "Previous Sub-Questions:\n" + "\n".join(parts) if parts else ""

    def _passage_dict_from_path(self, p: Dict[str, Any], source: str) -> Dict[str, Any]:
        title = p["title"]
        return {
            "title": title,
            "original_passage": self.get_original_passage(title),
            "score": p.get("score"),
            "matched_path": p.get("key_path"),
            "matched_value": p.get("value"),
            "dense_score": p.get("dense_score"),
            "bm25_score": p.get("bm25_score"),
            "dense_rank": p.get("dense_rank"),
            "bm25_rank": p.get("bm25_rank"),
            "source": source,
        }

    async def _retrieve_global_titles(self, query: str, top_k_titles: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # Retrieve more paths than titles to allow de-dup by title
        paths = await self.retriever.search_hybrid(query, top_k=top_k_titles * 5)
        passages: List[Dict[str, Any]] = []
        seen_titles: Set[str] = set()
        for p in paths:
            title = p["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            passages.append(self._passage_dict_from_path(p, source="global"))
            if len(passages) >= top_k_titles:
                break

        info = {
            "mode": "global",
            "query": query,
            "requested_titles": top_k_titles,
            "returned_titles": len(passages),
        }
        return passages, info

    def _union_dependency_pool(self, depends_on: List[str]) -> Set[int]:
        pool: Set[int] = set()
        for dep_id in depends_on:
            pool |= self.sq_pools.get(dep_id, set())
        return pool

    async def _retrieve_from_pool(self, query: str, pool_indices: Set[int], top_k_titles: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not pool_indices:
            return [], {"mode": "pool", "query": query, "pool_size": 0, "returned_titles": 0}

        scored_paths = await self.retriever.score_candidates_rrf(query, list(pool_indices), top_k=top_k_titles * 50)

        passages: List[Dict[str, Any]] = []
        seen_titles: Set[str] = set()
        for p in scored_paths:
            title = p["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            passages.append(self._passage_dict_from_path(p, source="pool"))
            if len(passages) >= top_k_titles:
                break

        info = {
            "mode": "pool",
            "query": query,
            "pool_size": len(pool_indices),
            "returned_titles": len(passages),
        }
        return passages, info

    def _top_titles_for_pool_update(self, passages: List[Dict[str, Any]], max_titles: int = 3) -> List[str]:
        titles: List[str] = []
        seen: Set[str] = set()
        for p in passages:
            t = p.get("title")
            if not t or t in seen:
                continue
            seen.add(t)
            titles.append(t)
            if len(titles) >= max_titles:
                break
        return titles

    def _expand_titles_to_pool_indices(self, seed_titles: List[str]) -> Tuple[Set[int], Dict[str, Any]]:
        candidate_indices: Set[int] = set()
        expanded_titles_seen: Set[str] = set()

        blocked_value_hits = 0
        total_values_seen = 0

        for title in seed_titles:
            values = self.linker.title_to_values.get(title, set())
            for v in values:
                total_values_seen += 1
                if v in self.linker.blocked_values:
                    blocked_value_hits += 1

            linked_info = self.linker.get_linked_info(title)
            for linked_title, shared_value in linked_info:
                expanded_titles_seen.add(linked_title)

                doc_indices = self.retriever.get_indices_for_title(linked_title)
                for idx in doc_indices:
                    path_val_raw = str(self.retriever.values[idx])
                    path_val_norm = self.linker.normalize_text(path_val_raw)
                    if shared_value in path_val_norm:
                        candidate_indices.add(idx)

        stats = {
            "seed_titles": seed_titles,
            "expanded_candidate_docs": len(expanded_titles_seen),
            "expanded_candidate_paths": len(candidate_indices),
            "blocked_values_total": len(self.linker.blocked_values),
            "blocked_value_hits": blocked_value_hits,
            "total_values_seen": total_values_seen,
            "max_docs_per_value": self.linker.max_docs_per_value,
        }
        return candidate_indices, stats

    async def _answer_with_passages(self, effective_query: str, main_query: str, previous_context: str, passages: List[Dict[str, Any]]) -> str:
        if not passages:
            return "Insufficient information."

        context_text = ""
        for i, p in enumerate(passages):
            context_text += f"Document {i+1} ({p['title']}):\n{p.get('original_passage','')}\n\n"

        prompt = (
            DETAILED_SUBQUESTION_ANSWERING_PROMPT.replace("{{main_query}}", main_query)
            .replace("{{subquestion}}", effective_query)
            .replace("{{passages}}", context_text)
            .replace("{{previous_context}}", previous_context)
        )

        response = await self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise question answering system. Give short, direct answers."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()

    async def answer_subquestion(self, sq: SubQuestion, decomposition: QueryDecomposition) -> None:
        previous_context = self._build_previous_context(sq, decomposition)
        effective_query = substitute_answers(sq.question, decomposition.subquestions)

        retrieval_info: Dict[str, Any] = {
            "effective_query": effective_query,
            "main_query": decomposition.main_query,
            "depends_on": list(sq.depends_on),
        }

        try:
            # Retrieval stage
            if not sq.depends_on:
                passages, info = await self._retrieve_global_titles(effective_query, self.top_k)
                retrieval_info["retrieval"] = info

                answer = await self._answer_with_passages(effective_query, decomposition.main_query, previous_context, passages)
                retrieval_info["answer_attempt"] = "global_top_k"

                # Fallback: broaden retrieval once if insufficient
                if _is_insufficient(answer) and self.top_k_fallback and self.top_k_fallback > self.top_k:
                    passages_fb, info_fb = await self._retrieve_global_titles(effective_query, self.top_k_fallback)
                    answer_fb = await self._answer_with_passages(effective_query, decomposition.main_query, previous_context, passages_fb)
                    retrieval_info["fallback"] = {
                        "used": True,
                        "retrieval": info_fb,
                        "answer_insufficient_before": True,
                    }
                    # Use fallback only if it improved from insufficient
                    if not _is_insufficient(answer_fb):
                        passages = passages_fb
                        answer = answer_fb
                        retrieval_info["fallback"]["adopted"] = True
                    else:
                        retrieval_info["fallback"]["adopted"] = False

            else:
                dep_pool = self._union_dependency_pool(sq.depends_on)
                retrieval_info["dep_pool_size"] = len(dep_pool)

                passages, info = await self._retrieve_from_pool(effective_query, dep_pool, self.top_k)
                retrieval_info["retrieval"] = info

                # Pool empty/weak -> fallback global
                if not passages:
                    retrieval_info["fallback"] = {"used": True, "reason": "empty_pool_or_no_hits"}
                    passages, info2 = await self._retrieve_global_titles(effective_query, self.top_k)
                    retrieval_info["retrieval"] = {"primary": info, "fallback_global": info2}

                answer = await self._answer_with_passages(effective_query, decomposition.main_query, previous_context, passages)

                # If still insufficient and we were pool-based, try global once
                if _is_insufficient(answer) and isinstance(retrieval_info.get("retrieval"), dict) and retrieval_info.get("dep_pool_size", 0) > 0:
                    passages2, info2 = await self._retrieve_global_titles(effective_query, self.top_k)
                    answer2 = await self._answer_with_passages(effective_query, decomposition.main_query, previous_context, passages2)
                    retrieval_info["fallback"] = {"used": True, "reason": "insufficient_from_pool", "fallback_global": info2}
                    if not _is_insufficient(answer2):
                        passages = passages2
                        answer = answer2
                        retrieval_info["fallback"]["adopted"] = True
                    else:
                        retrieval_info["fallback"]["adopted"] = False

            # Save answer + passages
            sq.answer = answer
            sq.retrieved_passages = passages

            # Pool update (B): expand from top-3 titles used for answering
            seed_titles = self._top_titles_for_pool_update(passages, max_titles=3)
            if _is_insufficient(answer):
                # If we couldn't answer, do not expand further to avoid polluting downstream
                self.sq_pools[sq.id] = set()
                retrieval_info["pool_update"] = {"skipped": True, "reason": "insufficient_answer"}
            else:
                pool_indices, pool_stats = self._expand_titles_to_pool_indices(seed_titles)
                self.sq_pools[sq.id] = pool_indices
                retrieval_info["pool_update"] = {"skipped": False, **pool_stats}

            sq.retrieval_info = retrieval_info

            # Logging
            context_text = ""
            for i, p in enumerate(passages):
                context_text += f"Document {i+1} ({p['title']}):\n{p.get('original_passage','')}\n\n"

            log_llm_call(
                call_type=f"SubQuestion Answering v8 ({sq.id})",
                input_text="OMITTED",
                output_text=answer,
                context={
                    "subquestion": effective_query,
                    "previous_context": previous_context,
                    "main_query": decomposition.main_query,
                    "passages": context_text,
                    "retrieval_info": retrieval_info,
                },
            )

        except Exception as e:
            log_llm_error(
                call_type=f"SubQuestion Answering v8 ({sq.id})",
                error=str(e),
                context={"subquestion": effective_query},
            )
            sq.answer = "Error generating answer."
            sq.retrieved_passages = []
            sq.retrieval_info = retrieval_info

    async def run(self, query: str) -> Dict[str, Any]:
        decomposition_result = await decompose_query(self.client, query)
        if not decomposition_result or not decomposition_result.get("success"):
            return {"error": "Decomposition failed", "predicted_answer": "Decomposition failed."}

        decomposition: QueryDecomposition = decomposition_result["decomposition"]

        execution_batches = get_execution_order(decomposition)
        for batch in execution_batches:
            for sq_id in batch:
                sq = decomposition.get_subquestion(sq_id)
                if sq:
                    await self.answer_subquestion(sq, decomposition)

        # Final synthesis uses union of SQ passages (unique titles)
        qa_history_parts: List[str] = []
        all_passages: List[Dict[str, Any]] = []
        seen_titles: Set[str] = set()

        for sq in decomposition.subquestions:
            if sq.answer:
                qa_history_parts.append(f"Q: {sq.question}\nA: {sq.answer}")
            for p in (sq.retrieved_passages or []):
                t = p.get("title")
                if t and t not in seen_titles:
                    seen_titles.add(t)
                    all_passages.append(p)

        context_text = "\n\n".join(
            [f"Document ({p['title']}):\n{p.get('original_passage','')}" for p in all_passages]
        )
        qa_history = "\n\n".join(qa_history_parts)

        final_prompt = (
            FINAL_ANSWER_SYNTHESIS_PROMPT.replace("{{main_question}}", query)
            .replace("{{subquestion_chain}}", qa_history)
            .replace("{{passages}}", context_text)
        )

        try:
            response = await self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise question answering system. Give short, direct answers."},
                    {"role": "user", "content": final_prompt},
                ],
                temperature=0.0,
            )
            final_answer = (response.choices[0].message.content or "").strip()

            log_llm_call(
                call_type="Final Synthesis v8",
                input_text="OMITTED",
                output_text=final_answer,
                context={
                    "main_question": query,
                    "subquestion_chain": qa_history,
                    "passages": context_text,
                },
            )

        except Exception as e:
            log_llm_error(call_type="Final Synthesis v8", error=str(e), context={"main_question": query})
            final_answer = "Error generating final answer."

        return {
            "decomposition": decomposition.to_dict(),
            "predicted_answer": final_answer,
            "passages": all_passages,
        }
