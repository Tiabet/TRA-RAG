#!/usr/bin/env python3
"""
New Multi-hop Pipeline v5
==========================
Retrieval Strategy:
1. Hybrid Search (BM25 + Dense) on Paths -> Top-k Paths -> Top-3 Initial Passages
2. Metadata Expansion: Find all passages linked to these Top-3 via metadata (Shared Value)
3. Path Reranking (Z-Score): 
   - Identify "Candidate Paths" that justify the expansion (paths containing the shared value).
   - Score these paths against the Sub-Question (SQ) using Z-Score Fusion of BM25 and Dense.
4. Select Final Top-3 Passages based on the best scoring paths.

Pre-requisites:
- Path embeddings and BM25 index must be ready (HybridPathRetriever).
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from openai import AsyncOpenAI

from query_decomposition import (
    decompose_query,
    QueryDecomposition,
    SubQuestion,
    substitute_answers,
    get_execution_order
)
from hybrid_path_retriever import HybridPathRetriever
from llm_logger import log_llm_call, log_llm_error

# Import prompts
from Prompt.answer import (
    DETAILED_SUBQUESTION_ANSWERING_PROMPT,
    FINAL_SUBQUESTION_ANSWERING_PROMPT
)
from Prompt.subquestion_answering_prompt import FINAL_ANSWER_SYNTHESIS_PROMPT


class MetadataLinkerV5:
    """Helper to find linked documents using metadata, returning shared values."""
    
    def __init__(self, metadata_path: str):
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        self.value_to_titles = defaultdict(set)
        self.title_to_values = defaultdict(set)
        self._build_graph()
        print(f"✓ Metadata Graph built: {len(self.title_to_values)} documents")

    def normalize_text(self, text):
        if text is None:
            return ""
        s = str(text).lower()
        s = s.replace(',', '')
        return s.strip()

    def _extract_values(self, obj, values_set):
        if isinstance(obj, dict):
            for k, v in obj.items():
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
            context_metadata = item.get('context_metadata', [])
            for doc in context_metadata:
                meta = doc.get('metadata', {})
                if not meta:
                    continue
                
                title = meta.get('title') or doc.get('title')
                if not title:
                    continue
                
                # Extract values
                values = set()
                values.add(self.normalize_text(title))
                
                self._extract_values(meta.get('attributes', {}), values)
                
                for rel in meta.get('relations', []):
                    if isinstance(rel, dict):
                        target = rel.get('target')
                        if target:
                            values.add(self.normalize_text(target))
                
                # Register
                for val in values:
                    self.value_to_titles[val].add(title)
                    self.title_to_values[title].add(val)

    def get_linked_info(self, title: str) -> List[Tuple[str, str]]:
        """
        Get all (linked_title, shared_value) pairs for the given title.
        """
        linked_info = []
        values = self.title_to_values.get(title, set())
        
        for val in values:
            # Get all docs sharing this value
            shared_docs = self.value_to_titles.get(val, set())
            for doc in shared_docs:
                if doc != title:
                    linked_info.append((doc, val))
            
        return linked_info


class NewMultihopPipelineV5:
    def __init__(
        self,
        client: AsyncOpenAI,
        retriever: HybridPathRetriever,
        linker: MetadataLinkerV5,
        hotpotqa_path: str,
        top_k: int = 3,
        verbose: bool = False
    ):
        self.client = client
        self.retriever = retriever
        self.linker = linker
        self.top_k = top_k
        self.verbose = verbose
        
        # Load original passages
        self.original_passages = self._load_original_passages(hotpotqa_path)
        if self.verbose:
            print(f"✓ Loaded {len(self.original_passages)} original passages")

    def _load_original_passages(self, hotpotqa_path: str) -> Dict[str, str]:
        with open(hotpotqa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        passages = {}
        for item in data:
            for title, sentences in item.get('context', []):
                if title not in passages:
                    passages[title] = ''.join(sentences).strip()
        return passages

    def get_original_passage(self, title: str) -> Optional[str]:
        return self.original_passages.get(title)

    async def retrieve_for_query(self, query: str) -> Tuple[List[Dict], Dict]:
        """
        V5 Retrieval Logic (Revised):
        1. Initial Retrieval: Get Top-K unique documents using Hybrid Path Search.
        2. Metadata Expansion: Find documents linked to Initial Top-K.
        3. Path Reranking (RRF): Score ONLY the expanded candidate paths.
        4. Selection: Select Top-K unique documents from Expanded set.
        5. Final Set: Initial Top-K + Expanded Top-K.
        """
        # 1. Initial Retrieval
        # Fetch more paths to ensure we get top_k unique documents
        initial_paths = await self.retriever.search_hybrid(query, top_k=self.top_k * 5)
        
        initial_passages = []
        initial_titles = set()
        
        for p in initial_paths:
            title = p['title']
            if title in initial_titles:
                continue
            
            initial_titles.add(title)
            original_passage = self.get_original_passage(title)
            
            initial_passages.append({
                'title': title,
                'original_passage': original_passage,
                'score': p['score'],
                'matched_path': p['key_path'],
                'matched_value': p['value'],
                'dense_score': p['dense_score'],
                'bm25_score': p['bm25_score'],
                'dense_rank': p.get('dense_rank'),
                'bm25_rank': p.get('bm25_rank'),
                'source': 'initial'
            })
            
            if len(initial_passages) >= self.top_k:
                break
        
        if self.verbose:
            print(f"  - Initial Retrieval: {list(initial_titles)}")

        # 2. Metadata Expansion & Candidate Path Identification
        candidate_indices = set()
        expanded_titles_seen = set()
        
        for title in initial_titles:
            # Get linked docs and the values that connect them
            linked_info = self.linker.get_linked_info(title)
            
            for linked_title, shared_value in linked_info:
                # Skip if it's already in initial set
                if linked_title in initial_titles:
                    continue
                
                expanded_titles_seen.add(linked_title)
                
                # Find paths in linked_title that contain shared_value
                # 1. Get all path indices for linked_title
                doc_indices = self.retriever.get_indices_for_title(linked_title)
                
                # 2. Filter by shared_value
                for idx in doc_indices:
                    path_val_raw = str(self.retriever.values[idx])
                    path_val_norm = self.linker.normalize_text(path_val_raw)
                    
                    # Check if shared_value is in the path value
                    if shared_value in path_val_norm:
                        candidate_indices.add(idx)

        if self.verbose:
            print(f"  - Expansion Candidates: {len(expanded_titles_seen)} docs, {len(candidate_indices)} paths")

        # 3. Path Reranking (RRF) on Expanded Candidates ONLY
        expanded_passages = []
        
        if candidate_indices:
            # We fetch more paths to ensure we can find top_k unique documents
            scored_expanded_paths = await self.retriever.score_candidates_rrf(
                query, 
                list(candidate_indices), 
                top_k=self.top_k * 5 
            )
            
            seen_expanded_titles = set()
            
            for p in scored_expanded_paths:
                title = p['title']
                # Ensure uniqueness against initial set and within expanded set
                if title in initial_titles or title in seen_expanded_titles:
                    continue
                
                seen_expanded_titles.add(title)
                original_passage = self.get_original_passage(title)
                
                expanded_passages.append({
                    'title': title,
                    'original_passage': original_passage,
                    'score': p['score'],
                    'matched_path': p['key_path'], # The path that justified selection
                    'matched_value': p['value'],
                    'dense_score': p['dense_score'],
                    'bm25_score': p['bm25_score'],
                    'dense_rank': p.get('dense_rank'),
                    'bm25_rank': p.get('bm25_rank'),
                    'source': 'expanded'
                })
                
                if len(expanded_passages) >= self.top_k:
                    break
        
        # 4. Combine Final Passages
        final_passages = initial_passages + expanded_passages
        
        stats = {
            "initial_count": len(initial_passages),
            "expanded_candidates_docs": len(expanded_titles_seen),
            "expanded_candidates_paths": len(candidate_indices),
            "expanded_selected": len(expanded_passages),
            "total_final": len(final_passages)
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
        # 1. Substitute dependencies
        context_for_query = self._build_simple_previous_context(sq, decomposition)
        effective_query = substitute_answers(sq.question, decomposition.subquestions)
        
        if self.verbose:
            print(f"\n[SQ: {sq.id}] {effective_query}")

        # 2. Retrieve (V5 Logic)
        passages, stats = await self.retrieve_for_query(effective_query)
        sq.retrieved_passages = passages
        sq.retrieval_info = stats

        # 3. Generate Answer
        if not passages:
            sq.answer = "Insufficient information."
            return

        context_text = ""
        for i, p in enumerate(passages):
            context_text += f"Document {i+1} ({p['title']}):\n{p['original_passage']}\n\n"

        prompt = DETAILED_SUBQUESTION_ANSWERING_PROMPT.replace(
            "{{main_query}}", decomposition.main_query
        ).replace(
            "{{subquestion}}", effective_query
        ).replace(
            "{{passages}}", context_text
        ).replace(
            "{{previous_context}}", context_for_query
        )

        try:
            response = await self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise question answering system. Give short, direct answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            answer = response.choices[0].message.content.strip()
            sq.answer = answer
            
            # Log the interaction
            log_llm_call(
                call_type=f"SubQuestion Answering ({sq.id})",
                input_text="OMITTED",
                output_text=answer,
                context={
                    "subquestion": effective_query,
                    "passages": context_text,
                    "previous_context": context_for_query,
                    "main_query": decomposition.main_query,
                    "retrieval_stats": stats
                }
            )
            
            if self.verbose:
                print(f"  -> Answer: {answer}")
                
        except Exception as e:
            print(f"Error answering subquestion: {e}")
            sq.answer = "Error generating answer."

    async def run(self, query: str) -> Dict:
        """Run the full pipeline."""
        if self.verbose:
            print(f"Processing Query: {query}")
            
        # 1. Decompose
        decomposition_result = await decompose_query(self.client, query)
        if not decomposition_result or not decomposition_result.get('success'):
            return {"error": "Decomposition failed", "predicted_answer": "Decomposition failed."}
            
        decomposition = decomposition_result['decomposition']

        if self.verbose:
            print("Decomposition:")
            for sq in decomposition.subquestions:
                print(f"  - {sq.id}: {sq.question}")

        # 2. Process Sub-questions
        execution_batches = get_execution_order(decomposition)
        
        for batch in execution_batches:
            for sq_id in batch:
                sq = decomposition.get_subquestion(sq_id)
                if sq:
                    await self.answer_subquestion(sq, decomposition)

        # 3. Final Synthesis
        final_context = []
        all_passages = []
        seen_titles = set()
        
        for sq in decomposition.subquestions:
            if sq.answer:
                final_context.append(f"Q: {sq.question}\nA: {sq.answer}")
            if sq.retrieved_passages:
                for p in sq.retrieved_passages:
                    if p['title'] not in seen_titles:
                        all_passages.append(p)
                        seen_titles.add(p['title'])
        
        context_text = "\n\n".join([f"Document ({p['title']}):\n{p['original_passage']}" for p in all_passages])
        qa_history = "\n\n".join(final_context)
        
        final_prompt = FINAL_ANSWER_SYNTHESIS_PROMPT.replace(
            "{{main_question}}", query
        ).replace(
            "{{subquestion_chain}}", qa_history
        ).replace(
            "{{passages}}", context_text
        )

        try:
            response = await self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise question answering system. Give short, direct answers."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.0
            )
            final_answer = response.choices[0].message.content.strip()
            
            # Log final synthesis
            log_llm_call(
                call_type="Final Synthesis",
                input_text="OMITTED",
                output_text=final_answer,
                context={
                    "main_question": query,
                    "subquestion_chain": qa_history,
                    "passages": context_text
                }
            )
            
        except Exception as e:
            final_answer = "Error generating final answer."

        return {
            "decomposition": decomposition.to_dict(),
            "predicted_answer": final_answer,
            "passages": all_passages
        }
