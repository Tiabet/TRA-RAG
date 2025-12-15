#!/usr/bin/env python3
"""
New Multi-hop Pipeline v4
==========================
Retrieval Strategy:
1. Hybrid Search (BM25 + Dense) on Paths -> Top-k Paths -> Top-3 Initial Passages
2. Metadata Expansion: Find all passages linked to these Top-3 via metadata
3. Passage Reranking: Calculate similarity between SQ and (Initial + Expanded) passages
4. Select Final Top-3 Passages for Answering

Pre-requisites:
- Passage embeddings must be pre-computed using generate_passage_embeddings.py
"""

import asyncio
import json
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Set
from pathlib import Path
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

# Import prompts
from Prompt.answer import (
    DETAILED_SUBQUESTION_ANSWERING_PROMPT,
    FINAL_SUBQUESTION_ANSWERING_PROMPT
)
from Prompt.subquestion_answering_prompt import FINAL_ANSWER_SYNTHESIS_PROMPT
from llm_logger import log_llm_call, log_llm_error


class MetadataLinker:
    """Helper to find linked documents using metadata."""
    
    def __init__(self, metadata_path: str):
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        self.value_to_titles = defaultdict(set)
        self.title_to_values = defaultdict(set)
        self._build_graph()
        print(f"✓ Metadata Graph built: {len(self.title_to_values)} documents")

    def _normalize_text(self, text):
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
            val = self._normalize_text(obj)
            if val:
                values_set.add(val)

    def _build_graph(self):
        for item in self.metadata:
            # Handle both list of docs (HotpotQA) and single doc structure if any
            # HotpotQA metadata structure: List of {question_id, context_metadata: [...]}
            # Or if it's just a list of docs?
            # Based on analyze_metadata_links.py:
            # for qa_item in metadata_data:
            #    context_metadata = qa_item.get('context_metadata', [])
            
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
                values.add(self._normalize_text(title))
                
                self._extract_values(meta.get('attributes', {}), values)
                
                for rel in meta.get('relations', []):
                    if isinstance(rel, dict):
                        target = rel.get('target')
                        if target:
                            values.add(self._normalize_text(target))
                
                # Register
                for val in values:
                    self.value_to_titles[val].add(title)
                    self.title_to_values[title].add(val)

    def get_linked_titles(self, title: str) -> Set[str]:
        """Get all titles linked to the given title via shared metadata values."""
        linked_titles = set()
        values = self.title_to_values.get(title, set())
        
        for val in values:
            # Get all docs sharing this value
            shared_docs = self.value_to_titles.get(val, set())
            linked_titles.update(shared_docs)
            
        # Remove self
        if title in linked_titles:
            linked_titles.remove(title)
            
        return linked_titles


class PassageReranker:
    """Reranks passages using pre-computed embeddings."""
    
    def __init__(self, embeddings_path: str, client: AsyncOpenAI):
        print(f"Loading passage embeddings from {embeddings_path}...")
        data = np.load(embeddings_path, allow_pickle=True)
        self.titles = data['titles']
        self.embeddings = data['embeddings']
        self.client = client
        
        # Map title -> index for fast lookup
        self.title_to_idx = {t: i for i, t in enumerate(self.titles)}
        print(f"✓ Loaded embeddings for {len(self.titles)} passages")

    async def rerank(self, query: str, candidate_titles: List[str], top_k: int = 3) -> List[Dict]:
        """
        Rerank candidate titles based on similarity to query.
        Returns list of {'title': str, 'score': float}
        """
        # 1. Embed Query
        response = await self.client.embeddings.create(
            input=query.replace("\n", " "),
            model="text-embedding-3-small"
        )
        query_vec = np.array(response.data[0].embedding, dtype=np.float32)
        # Normalize
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
            
        # 2. Gather Candidate Embeddings
        candidate_indices = []
        valid_titles = []
        
        for t in candidate_titles:
            if t in self.title_to_idx:
                candidate_indices.append(self.title_to_idx[t])
                valid_titles.append(t)
        
        if not candidate_indices:
            return []
            
        candidate_matrix = self.embeddings[candidate_indices]
        
        # 3. Compute Similarity (Dot product since normalized)
        scores = np.dot(candidate_matrix, query_vec)
        
        # 4. Sort
        results = []
        for i, score in enumerate(scores):
            results.append({
                'title': valid_titles[i],
                'score': float(score)
            })
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]


class NewMultihopPipelineV4:
    """
    Pipeline V4: Hybrid Retrieval -> Metadata Expansion -> Embedding Reranking -> Answer
    """
    
    def __init__(
        self,
        client: AsyncOpenAI,
        retriever: HybridPathRetriever,
        linker: MetadataLinker,
        reranker: PassageReranker,
        hotpotqa_path: str = 'HotpotQA/hotpotqa_sample_200.json',
        top_k: int = 3,
        verbose: bool = True
    ):
        self.client = client
        self.retriever = retriever
        self.linker = linker
        self.reranker = reranker
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

    async def retrieve_for_query(self, query: str) -> List[Dict]:
        """
        V4 Retrieval Logic:
        1. Get Top-3 from Hybrid Retriever (Initial Set)
        2. Expand using Metadata Links
        3. Rerank all candidates using Passage Embeddings
        """
        # 1. Initial Retrieval (Fetch more to ensure we get top_k unique titles)
        paths = await self.retriever.search_hybrid(query, top_k=self.top_k * 5)
        
        initial_titles = set()
        provenance_map = {} # Store metadata for each title

        for p in paths:
            title = p['title']
            if title not in provenance_map:
                 provenance_map[title] = {
                     'matched_path': p.get('matched_path', 'initial_retrieval'),
                     'matched_value': p.get('matched_value', 'N/A')
                 }
            
            initial_titles.add(title)
            if len(initial_titles) >= self.top_k:
                break
        
        if self.verbose:
            print(f"  - Initial Retrieval: {list(initial_titles)}")

        # 2. Metadata Expansion
        expanded_titles = set(initial_titles)
        for title in initial_titles:
            linked = self.linker.get_linked_titles(title)
            expanded_titles.update(linked)
            
        if self.verbose:
            print(f"  - After Expansion: {len(expanded_titles)} documents")

        # 3. Reranking
        if self.verbose:
            print(f"  - Reranking {len(expanded_titles)} candidates against query: '{query}'")
            
        reranked_results = await self.reranker.rerank(query, list(expanded_titles), top_k=self.top_k)
        reranked_titles = [r['title'] for r in reranked_results]
        
        if self.verbose:
            print(f"  - Top {self.top_k} after Reranking: {reranked_titles}")

        # 4. Union Strategy: Initial Top-k + Reranked Top-k
        final_titles_set = set(initial_titles)
        final_titles_set.update(reranked_titles)
        
        if self.verbose:
            print(f"  - Final Union Set ({len(final_titles_set)}): {list(final_titles_set)}")

        # 5. Format Results
        final_passages = []
        # Map title -> score from reranked results for reference
        title_to_score = {r['title']: r['score'] for r in reranked_results}
        
        for title in final_titles_set:
            original_passage = self.get_original_passage(title)
            score = title_to_score.get(title, 0.0) # 0.0 if not in top-k reranked
            
            # Retrieve provenance info
            prov = provenance_map.get(title, {
                'matched_path': "expanded_or_reranked", # Default if not in initial map
                'matched_value': "N/A"
            })

            final_passages.append({
                'title': title,
                'original_passage': original_passage,
                'score': score,
                'matched_path': prov['matched_path'],
                'matched_value': prov['matched_value']
            })
            
        return final_passages

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

        # 2. Retrieve (V4 Logic)
        passages = await self.retrieve_for_query(effective_query)
        sq.retrieved_passages = passages # Store for analysis

        # 3. Generate Answer
        if not passages:
            sq.answer = "Insufficient information."
            return

        # Prepare context text
        context_text = ""
        for i, p in enumerate(passages, 1):
            context_text += f"Passage {i} (Title: {p['title']}):\n{p['original_passage']}\n\n"

        # Prompt
        prompt = DETAILED_SUBQUESTION_ANSWERING_PROMPT.replace(
            "{{main_query}}", decomposition.main_query
        ).replace(
            "{{subquestion}}", effective_query
        ).replace(
            "{{passages}}", context_text
        ).replace(
            "{{previous_context}}", context_for_query
        )

        response = await self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        sq.answer = response.choices[0].message.content.strip()
        if self.verbose:
            print(f"  -> Answer: {sq.answer}")

    async def run(self, query: str) -> Dict:
        start_time = asyncio.get_event_loop().time()
        
        # 1. Decompose
        if self.verbose:
            print(f"\nMain Query: {query}")
            print("Decomposing...")
        
        decomp_result = await decompose_query(self.client, query)
        if not decomp_result['success']:
            return {
                'question': query,
                'predicted_answer': "Decomposition failed.",
                'decomposition': {},
                'error': decomp_result.get('error'),
                'total_time': asyncio.get_event_loop().time() - start_time
            }
            
        decomposition = decomp_result['decomposition']
        execution_batches = get_execution_order(decomposition)
        
        # 2. Execute SQs
        for batch in execution_batches:
            for sq_id in batch:
                sq = decomposition.get_subquestion(sq_id)
                await self.answer_subquestion(sq, decomposition)
            
        # 3. Final Answer
        final_context = ""
        # Collect all unique passages used
        seen_titles = set()
        all_passages = []
        for sq in decomposition.subquestions:
            if sq.retrieved_passages:
                for p in sq.retrieved_passages:
                    if p['title'] not in seen_titles:
                        seen_titles.add(p['title'])
                        all_passages.append(p)
        
        for i, p in enumerate(all_passages, 1):
            final_context += f"Passage {i} (Title: {p['title']}):\n{p['original_passage']}\n\n"
            
        qa_pairs = []
        for sq in decomposition.subquestions:
            qa_pairs.append(f"Q: {sq.question}\nA: {sq.answer}")
        qa_history = "\n\n".join(qa_pairs)
        
        final_prompt = FINAL_ANSWER_SYNTHESIS_PROMPT.replace(
            "{{main_question}}", query
        ).replace(
            "{{subquestion_chain}}", qa_history
        ).replace(
            "{{passages}}", final_context
        )
        
        response = await self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0
        )
        
        final_answer = response.choices[0].message.content.strip()
        total_time = asyncio.get_event_loop().time() - start_time
        
        return {
            'question': query,
            'predicted_answer': final_answer,
            'decomposition': decomposition.to_dict(),
            'total_time': total_time
        }

# Test runner
async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    
    # Setup
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    client = AsyncOpenAI(api_key=os.getenv('ALICE_OPENAI_KEY'), base_url=os.getenv('ALICE_EMBED_URL'))
    
    # Initialize Components
    retriever = HybridPathRetriever()
    
    # Paths
    hotpot_path = 'HotpotQA/hotpotqa_sample_200.json'
    metadata_path = 'HotpotQA/hotpotqa_sample_200_metadata.json'
    embeddings_path = 'HotpotQA/passage_embeddings.npz'
    
    linker = MetadataLinker(metadata_path)
    reranker = PassageReranker(embeddings_path, client)
    
    pipeline = NewMultihopPipelineV4(
        client=client,
        retriever=retriever,
        linker=linker,
        reranker=reranker,
        hotpotqa_path=hotpot_path,
        top_k=3
    )
    
    # Load Data
    with open(hotpot_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    results = []
    for item in data[:args.limit]:
        res = await pipeline.run(item['question'])
        res['id'] = item['_id']
        res['gold_answer'] = item['answer']
        results.append(res)
        
    # Save
    with open('Results/test_v4_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
