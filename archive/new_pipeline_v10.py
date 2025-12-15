#!/usr/bin/env python3
"""Pipeline v10 (No QD) — value-only 1-hop expansion

Design (per user request)
-------------------------
1) Initial retrieval: main_query -> top-10 RRF *paths* (no title-unique constraint)
2) Seed values: extract primitive values from those 10 paths' `value` fields
3) 1-hop neighborhood: link documents that share ANY of those seed values
   (value-only linking; no key linking; no filtering)
4) Candidate paths: all path indices for neighborhood titles
5) Final selection: RRF score candidates by main_query -> keep top-5 passages (title-unique)
6) Answer: single-shot answer using those top-5 passages

Notes
-----
- "value-only" here means we only index primitive metadata values (strings/numbers/bools)
  from attributes + relations; not metadata keys.
- We attempt to parse JSON-like strings in path values to extract primitives.
"""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import AsyncOpenAI

from hybrid_path_retriever import HybridPathRetriever
from llm_logger import log_llm_call, log_llm_error

from Prompt.answer import DETAILED_SUBQUESTION_ANSWERING_PROMPT


def _normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).lower().strip()
    s = s.replace(",", "")
    return s


def _extract_primitives(obj: Any, out: Set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _extract_primitives(v, out)
    elif isinstance(obj, list):
        for it in obj:
            _extract_primitives(it, out)
    elif isinstance(obj, (str, int, float, bool)):
        t = _normalize_text(obj)
        if t:
            out.add(t)


def _extract_primitives_from_maybe_json_string(s: str) -> Set[str]:
    out: Set[str] = set()
    if s is None:
        return out
    raw = str(s).strip()

    # Try JSON parse for strings that look like JSON
    try:
        if raw.startswith("{") or raw.startswith("["):
            data = json.loads(raw)
            _extract_primitives(data, out)
            return out
    except Exception:
        pass

    # Fallback: treat as plain string
    t = _normalize_text(raw)
    if t:
        out.add(t)
    return out


class MetadataLinkerValuesOnly:
    """Unfiltered value-only linker (no keys)."""

    def __init__(self, metadata_path: str):
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.value_to_titles = defaultdict(set)  # value_token -> titles
        self.title_to_values = defaultdict(set)  # title -> value_tokens

        self._build_graph()
        print(
            f"✓ Metadata Graph built (VALUES ONLY): {len(self.title_to_values)} docs, {len(self.value_to_titles)} values"
        )

    def _add_value(self, title: str, val: Any):
        t = _normalize_text(val)
        if not t:
            return
        self.value_to_titles[t].add(title)
        self.title_to_values[title].add(t)

    def _build_graph(self):
        for item in self.metadata:
            for doc in item.get("context_metadata", []) or []:
                meta = doc.get("metadata", {})
                if not meta:
                    continue

                title = meta.get("title") or doc.get("title")
                if not title:
                    continue

                # Treat title itself as a value token as well
                self._add_value(title, title)

                # attributes values
                prims: Set[str] = set()
                _extract_primitives(meta.get("attributes", {}), prims)
                for v in prims:
                    self._add_value(title, v)

                # relations values (both relation label and target are values in metadata)
                prims = set()
                _extract_primitives(meta.get("relations", []), prims)
                for v in prims:
                    self._add_value(title, v)

    def get_neighbor_titles_by_seed_values(self, seed_values: Set[str]) -> Set[str]:
        neighbors: Set[str] = set()
        for v in seed_values:
            for t in self.value_to_titles.get(v, set()):
                neighbors.add(t)
        return neighbors


class PipelineV10:
    def __init__(
        self,
        client: AsyncOpenAI,
        retriever: HybridPathRetriever,
        linker: MetadataLinkerValuesOnly,
        dataset_path: str,
        initial_top_paths: int = 10,
        final_top_passages: int = 5,
        verbose: bool = False,
    ):
        self.client = client
        self.retriever = retriever
        self.linker = linker
        self.dataset_path = dataset_path
        self.initial_top_paths = initial_top_paths
        self.final_top_passages = final_top_passages
        self.verbose = verbose

        self.original_passages = self._load_original_passages(dataset_path)

    def _load_original_passages(self, dataset_path: str) -> Dict[str, str]:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        passages: Dict[str, str] = {}
        for item in data:
            for title, sentences in item.get("context", []):
                if title not in passages:
                    passages[title] = "".join(sentences).strip()
        return passages

    def get_original_passage(self, title: str) -> Optional[str]:
        return self.original_passages.get(title)

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

    async def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # 1) Initial top-N paths (no unique constraint)
        initial_paths = await self.retriever.search_hybrid(query, top_k=self.initial_top_paths)

        # 2) Seed values extracted from those path values
        seed_values: Set[str] = set()
        for p in initial_paths:
            for v in _extract_primitives_from_maybe_json_string(p.get("value", "")):
                seed_values.add(v)

        # 3) Neighborhood titles linked by seed values only
        neighborhood_titles = self.linker.get_neighbor_titles_by_seed_values(seed_values)

        # 4) Candidate paths are all paths for neighborhood titles
        candidate_indices: Set[int] = set()
        for t in neighborhood_titles:
            for idx in self.retriever.get_indices_for_title(t):
                candidate_indices.add(idx)

        # 5) RRF within candidate paths, then keep top-K passages (title-unique)
        scored = await self.retriever.score_candidates_rrf(query, list(candidate_indices), top_k=self.final_top_passages * 50)

        passages: List[Dict[str, Any]] = []
        seen_titles: Set[str] = set()
        for p in scored:
            title = p["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            passages.append(self._passage_dict_from_path(p, source="value_neighborhood_rrf"))
            if len(passages) >= self.final_top_passages:
                break

        info = {
            "initial_top_paths": self.initial_top_paths,
            "final_top_passages": self.final_top_passages,
            "initial_paths": [
                {"title": p.get("title"), "key_path": p.get("key_path"), "value": p.get("value"), "score": p.get("score")}
                for p in initial_paths
            ],
            "seed_values_count": len(seed_values),
            "seed_values_sample": list(sorted(seed_values))[:30],
            "neighborhood_titles": len(neighborhood_titles),
            "candidate_paths": len(candidate_indices),
            "final_selected_titles": len(passages),
        }

        return passages, info

    async def answer(self, query: str, passages: List[Dict[str, Any]]) -> str:
        if not passages:
            return "Insufficient information."

        context_text = ""
        for i, p in enumerate(passages):
            context_text += f"Document {i+1} ({p['title']}):\n{p.get('original_passage','')}\n\n"

        prompt = (
            DETAILED_SUBQUESTION_ANSWERING_PROMPT.replace("{{main_query}}", query)
            .replace("{{subquestion}}", query)
            .replace("{{passages}}", context_text)
            .replace("{{previous_context}}", "")
        )

        resp = await self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise question answering system. Give short, direct answers."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    async def run(self, query: str) -> Dict[str, Any]:
        try:
            passages, retrieval_info = await self.retrieve(query)
            predicted = await self.answer(query, passages)

            log_llm_call(
                call_type="Pipeline v10 Answer",
                input_text="OMITTED",
                output_text=predicted,
                context={
                    "main_question": query,
                    "retrieval_info": retrieval_info,
                    "passages": "\n\n".join([f"{p['title']}\n{p.get('original_passage','')}" for p in passages]),
                },
            )

            return {
                "question": query,
                "predicted_answer": predicted,
                "retrieved_passages": passages,
                "retrieval_info": retrieval_info,
            }

        except Exception as e:
            log_llm_error(call_type="Pipeline v10", error=str(e), context={"main_question": query})
            return {"question": query, "predicted_answer": "Error", "error": str(e)}
