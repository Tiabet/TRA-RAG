#!/usr/bin/env python3
"""New Multi-hop Pipeline v7 (QD_v2)

What changes vs v7
------------------
- Keeps v7 retrieval/expansion behavior (SQ initial, main_query expanded rerank + filters).
- Adds an extra LLM step per SQ to select which passages were actually needed
  to justify the SQ answer.
- Final synthesis uses ONLY the selected passages across all SQs (unique titles).

Why
---
SQ answering typically needs only 2-3 docs; passing all retrieved docs into final
synthesis can add noise. This variant tries to reduce noise by keeping only
"needed" passages.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

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
from Prompt.passage_selection_prompt import SUBQUESTION_ANSWERING_WITH_SELECTION_PROMPT

from new_multihop_pipeline_v7 import MetadataLinkerV7
from qd_v2 import QueryDecompositionV2, SubQuestionV2


def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _safe_parse_answer_and_needed_docs(
    text: str, max_doc_num: int
) -> Tuple[Optional[str], List[int], Dict[str, Any]]:
    """Parse LLM JSON for answer + needed_docs.

    Returns:
        (answer_or_none, needed_docs, selection_info)
    """
    selection_info: Dict[str, Any] = {"raw": text}
    try:
        json_str = _extract_json_object(text) or text
        data = json.loads(json_str)

        answer = data.get("answer")
        if isinstance(answer, str):
            answer = answer.strip()
        else:
            answer = None

        needed = data.get("needed_docs", [])
        if not isinstance(needed, list):
            needed = []

        needed_ints: List[int] = []
        for x in needed:
            try:
                n = int(x)
                if 1 <= n <= max_doc_num:
                    needed_ints.append(n)
            except Exception:
                continue

        seen = set()
        needed_ints = [n for n in needed_ints if not (n in seen or seen.add(n))]

        selection_info.update({"parsed": data})
        return answer, needed_ints, selection_info
    except Exception as e:
        selection_info["error"] = str(e)
        return None, [], selection_info


class NewMultihopPipelineV7_QDv2:
    def __init__(
        self,
        client: AsyncOpenAI,
        retriever: HybridPathRetriever,
        linker: MetadataLinkerV7,
        hotpotqa_path: str,
        top_k: int = 3,
        verbose: bool = False,
        passage_selector_model: str = "openai/gpt-4o-mini",
    ):
        self.client = client
        self.retriever = retriever
        self.linker = linker
        self.top_k = top_k
        self.verbose = verbose
        self.passage_selector_model = passage_selector_model

        self.original_passages = self._load_original_passages(hotpotqa_path)

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
        """Initial retrieval by SQ; expansion reranking by main_query (same as v7)."""

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
        parts: List[str] = []
        for dep_id in current_sq.depends_on:
            dep_sq = decomposition.get_subquestion(dep_id)
            if dep_sq and dep_sq.answer:
                parts.append(f"{dep_id}: {dep_sq.question}")
                parts.append(f"Answer: {dep_sq.answer}")
                parts.append("")
        return "Previous Sub-Questions:\n" + "\n".join(parts) if parts else ""

    def _format_documents_numbered(self, passages: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for i, p in enumerate(passages, start=1):
            title = p.get("title", "")
            txt = p.get("original_passage", "") or ""
            lines.append(f"Document {i} (Title: {title}):\n{txt}\n")
        return "\n".join(lines)

    async def answer_and_select_passages(self, subquestion: str, passages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Single-call: produce answer + minimal supporting docs."""
        if not passages:
            return "Insufficient information.", [], {"needed_docs": [], "reason": "no_passages"}

        documents_text = self._format_documents_numbered(passages)
        prompt = (
            SUBQUESTION_ANSWERING_WITH_SELECTION_PROMPT.replace("{{subquestion}}", subquestion)
            .replace("{{documents}}", documents_text)
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.passage_selector_model,
                messages=[
                    {"role": "system", "content": "Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            out = (response.choices[0].message.content or "").strip()

            answer, needed_docs, selection_info = _safe_parse_answer_and_needed_docs(out, max_doc_num=len(passages))

            # If parsing fails, fall back conservatively.
            if not answer:
                selection_info["fallback"] = "kept_all_unparsed"
                return out or "Insufficient information.", passages, selection_info

            # If model says insufficient, keep none.
            if answer.lower().startswith("insufficient information"):
                selection_info["needed_docs"] = []
                return answer, [], selection_info

            selected = [passages[i - 1] for i in needed_docs]
            selection_info["needed_docs"] = needed_docs
            selection_info["needed_titles"] = [p.get("title") for p in selected]

            # Safety fallback: avoid accidentally dropping all evidence when answer is non-empty.
            if not selected:
                selection_info["fallback"] = "kept_all_empty_selection"
                return answer, passages, selection_info

            return answer, selected, selection_info

        except Exception as e:
            # Fallback: keep all passages and mark error
            return "Error generating answer.", passages, {"error": str(e), "fallback": "kept_all"}

    async def answer_subquestion(self, sq: SubQuestion, decomposition: QueryDecomposition) -> None:
        context_for_query = self._build_simple_previous_context(sq, decomposition)
        effective_query = substitute_answers(sq.question, decomposition.subquestions)

        passages, stats = await self.retrieve_for_query(effective_query, decomposition.main_query)
        sq.retrieved_passages = passages
        sq.retrieval_info = stats

        # Single-call: answer + select
        answer, selected_passages, selection_info = await self.answer_and_select_passages(
            subquestion=effective_query,
            passages=passages,
        )
        sq.answer = answer
        sq.selected_passages = selected_passages  # type: ignore[attr-defined]
        sq.selection_info = selection_info  # type: ignore[attr-defined]

        # For logging, include the full passages that were available (not just selected)
        context_text = ""
        for i, p in enumerate(passages):
            context_text += f"Document {i+1} ({p['title']}):\n{p.get('original_passage','')}\n\n"

        log_llm_call(
            call_type=f"SubQuestion Answering+Select (single-call) ({sq.id})",
            input_text="OMITTED",
            output_text=answer,
            context={
                "subquestion": effective_query,
                "passages": context_text,
                "previous_context": context_for_query,
                "main_query": decomposition.main_query,
                "retrieval_stats": stats,
                "selection_info": selection_info,
            },
        )

    def _to_qd_v2(self, decomposition: QueryDecomposition) -> QueryDecompositionV2:
        subqs_v2: List[SubQuestionV2] = []
        for sq in decomposition.subquestions:
            selected = getattr(sq, "selected_passages", [])
            selection_info = getattr(sq, "selection_info", {})
            subqs_v2.append(
                SubQuestionV2(
                    id=sq.id,
                    question=sq.question,
                    depends_on=sq.depends_on,
                    reasoning=sq.reasoning,
                    answer=sq.answer,
                    retrieved_passages=sq.retrieved_passages,
                    retrieval_info=sq.retrieval_info,
                    selected_passages=selected,
                    selection_info=selection_info,
                )
            )
        return QueryDecompositionV2(
            main_query=decomposition.main_query,
            question_type=decomposition.question_type,
            reasoning=decomposition.reasoning,
            subquestions=subqs_v2,
        )

    async def run(self, query: str) -> Dict[str, Any]:
        decomposition_result = await decompose_query(self.client, query)
        if not decomposition_result or not decomposition_result.get("success"):
            return {"error": "Decomposition failed", "predicted_answer": "Decomposition failed."}

        decomposition = decomposition_result["decomposition"]

        execution_batches = get_execution_order(decomposition)
        for batch in execution_batches:
            for sq_id in batch:
                sq = decomposition.get_subquestion(sq_id)
                if sq:
                    await self.answer_subquestion(sq, decomposition)

        # Final synthesis: use ONLY selected passages
        all_passages: List[Dict[str, Any]] = []
        seen_titles = set()
        qa_history_parts: List[str] = []

        for sq in decomposition.subquestions:
            if sq.answer:
                qa_history_parts.append(f"Q: {sq.question}\nA: {sq.answer}")

            selected = getattr(sq, "selected_passages", None)
            if selected is None:
                selected = sq.retrieved_passages

            for p in selected or []:
                title = p.get("title")
                if title and title not in seen_titles:
                    all_passages.append(p)
                    seen_titles.add(title)

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
                    {
                        "role": "system",
                        "content": "You are a precise question answering system. Give short, direct answers.",
                    },
                    {"role": "user", "content": final_prompt},
                ],
                temperature=0.0,
            )
            final_answer = (response.choices[0].message.content or "").strip()

            log_llm_call(
                call_type="Final Synthesis (QD_v2 selected passages)",
                input_text="OMITTED",
                output_text=final_answer,
                context={
                    "main_question": query,
                    "subquestion_chain": qa_history,
                    "passages": context_text,
                    "selected_passage_titles": [p.get("title") for p in all_passages],
                },
            )

        except Exception as e:
            log_llm_error(call_type="Final Synthesis", error=str(e), context={"main_question": query})
            final_answer = "Error generating final answer."

        qd_v2 = self._to_qd_v2(decomposition)

        return {
            "decomposition": qd_v2.to_dict(),
            "predicted_answer": final_answer,
            "passages": all_passages,
        }
