#!/usr/bin/env python3
"""Pipeline v9 (No QD)

Experiment
----------
- Initial: retrieve top-3 RRF paths for main query
- Expansion: collect 1-hop neighborhood via FULL metadata linking
  (no shared value/key filtering, no degree cutoff)
- Rerank: RRF score ONLY paths belonging to neighborhood titles, then keep top-10 passages
- Answer: single-shot answer with those passages

Notes
-----
"Full linking" here means we index BOTH keys and primitive values in metadata,
including relation labels and targets.
"""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import AsyncOpenAI

from hybrid_path_retriever import HybridPathRetriever
from llm_logger import log_llm_call, log_llm_error

from Prompt.answer_prompt import DETAILED_SUBQUESTION_ANSWERING_PROMPT


class MetadataLinkerFull:
    """Metadata linker for v9: VALUE-only linking.

    Historical note: the first v9 attempt indexed BOTH keys and values.
    That makes structural keys like "relation" / "target" act as giant hubs,
    exploding the 1-hop neighborhood to near-corpus size.

    Current behavior (as requested): index primitive VALUES only (including
    relation labels and targets), not metadata keys.
    """

    def __init__(self, metadata_path: str):
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.value_to_titles = defaultdict(set)  # token -> titles
        self.title_to_values = defaultdict(set)  # title -> tokens

        self._build_graph()
        print(f"✓ Metadata Graph built (FULL): {len(self.title_to_values)} docs, {len(self.value_to_titles)} tokens")

    def normalize_text(self, text: Any) -> str:
        if text is None:
            return ""
        s = str(text).lower().strip()
        s = s.replace(",", "")
        return s

    def _extract_tokens(self, obj: Any, tokens: Set[str]):
        if isinstance(obj, dict):
            for _, v in obj.items():
                self._extract_tokens(v, tokens)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_tokens(item, tokens)
        elif isinstance(obj, (str, int, float, bool)):
            val_tok = self.normalize_text(obj)
            if val_tok:
                tokens.add(val_tok)

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

                tokens: Set[str] = set()
                tokens.add(self.normalize_text(title))

                # VALUE-only: include primitive values from attributes
                self._extract_tokens(meta.get("attributes", {}), tokens)

                # VALUE-only: include primitive values from relations (incl. relation labels + targets)
                self._extract_tokens(meta.get("relations", []), tokens)

                for tok in tokens:
                    if not tok:
                        continue
                    self.value_to_titles[tok].add(title)
                    self.title_to_values[title].add(tok)

    def get_neighbor_titles_1hop(self, seed_titles: List[str]) -> Set[str]:
        neighbors: Set[str] = set()
        for title in seed_titles:
            neighbors.add(title)
            for tok in self.title_to_values.get(title, set()):
                for other_title in self.value_to_titles.get(tok, set()):
                    neighbors.add(other_title)
        return neighbors


class PipelineV9:
    def __init__(
        self,
        client: AsyncOpenAI,
        retriever: HybridPathRetriever,
        linker: MetadataLinkerFull,
        dataset_path: str,
        initial_top_paths: int = 3,
        final_top_passages: int = 10,
        verbose: bool = False,
    ):
        self.client = client
        self.retriever = retriever
        self.linker = linker
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
        # 1) Initial top-3 paths (no title dedup; seeds will dedup)
        initial_paths = await self.retriever.search_hybrid(query, top_k=self.initial_top_paths)
        seed_titles: List[str] = []
        seen_seed: Set[str] = set()
        for p in initial_paths:
            t = p.get("title")
            if t and t not in seen_seed:
                seen_seed.add(t)
                seed_titles.append(t)

        # 2) 1-hop neighborhood titles via FULL linking
        neighborhood_titles = self.linker.get_neighbor_titles_1hop(seed_titles)

        # 3) Candidate path indices are all paths for neighborhood titles
        candidate_indices: Set[int] = set()
        for t in neighborhood_titles:
            for idx in self.retriever.get_indices_for_title(t):
                candidate_indices.add(idx)

        # 4) RRF within candidate paths, then keep top-10 unique titles
        scored = await self.retriever.score_candidates_rrf(query, list(candidate_indices), top_k=self.final_top_passages * 50)

        passages: List[Dict[str, Any]] = []
        seen_titles: Set[str] = set()
        for p in scored:
            title = p["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            passages.append(self._passage_dict_from_path(p, source="neighborhood_rrf"))
            if len(passages) >= self.final_top_passages:
                break

        info = {
            "initial_top_paths": self.initial_top_paths,
            "initial_paths": [{"title": p.get("title"), "key_path": p.get("key_path"), "value": p.get("value"), "score": p.get("score")} for p in initial_paths],
            "seed_titles": seed_titles,
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
                call_type="Pipeline v9 Answer",
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
            log_llm_error(call_type="Pipeline v9", error=str(e), context={"main_question": query})
            return {"question": query, "predicted_answer": "Error", "error": str(e)}
