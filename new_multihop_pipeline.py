#!/usr/bin/env python3
"""
New Multi-hop Pipeline v2
==========================
Simplified pipeline using hybrid path retrieval.

Pipeline:
1. Query Decomposition (existing)
2. For each SQ:
   - Hybrid Search (BM25 + Dense) → Top-k paths
   - Path → Original Passage mapping
   - SQ Answering (passage-based)
3. Final Answer: Main Query 답변 (모든 SQ의 unique passages 사용)
"""

import asyncio
import json
import time
import sqlite3
from typing import Dict, List, Optional
from pathlib import Path
from openai import AsyncOpenAI

from query_decomposition import (
    decompose_query,
    QueryDecomposition,
    SubQuestion,
    substitute_answers,
    get_execution_order
)
from hybrid_path_retriever import HybridPathRetriever

# Import prompts from Prompt folder (same as sequential_answering.py)
from Prompt.answer import (
    DETAILED_SUBQUESTION_ANSWERING_PROMPT,
    FINAL_SUBQUESTION_ANSWERING_PROMPT
)
from Prompt.subquestion_answering_prompt import FINAL_ANSWER_SYNTHESIS_PROMPT


class NewMultihopPipeline:
    """New pipeline using hybrid path retrieval."""
    
    def __init__(
        self,
        client: AsyncOpenAI,
        retriever: HybridPathRetriever,
        db_path: str = 'HotpotQA/metadata_v3.db',
        top_k: int = 3,
        verbose: bool = True
    ):
        """
        Args:
            client: AsyncOpenAI client for LLM calls
            retriever: HybridPathRetriever instance
            db_path: Path to metadata database (for full metadata lookup)
            top_k: Number of paths to retrieve per query
            verbose: Print progress
        """
        self.client = client
        self.retriever = retriever
        self.db_path = db_path
        self.top_k = top_k
        self.verbose = verbose
        
        # Connect to database for full metadata lookup
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def get_full_metadata(self, title: str) -> Optional[Dict]:
        """Get full metadata for a title from database."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT metadata_json FROM metadata WHERE title = ?",
            (title,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row['metadata_json'])
        return None
    
    async def retrieve_for_query(self, query: str) -> List[Dict]:
        """
        Retrieve passages for a query using hybrid search.
        Ensures top_k unique titles are returned.
        
        Args:
            query: The query text
            
        Returns:
            List of passage dicts with title and full metadata (top_k unique titles)
        """
        # Request more paths to ensure we get enough unique titles
        # Fetch up to 10x top_k paths to find enough unique titles
        fetch_k = self.top_k * 10
        paths = await self.retriever.search_hybrid(query, top_k=fetch_k)
        
        # Get unique titles until we have top_k
        seen_titles = set()
        passages = []
        
        for path in paths:
            if len(passages) >= self.top_k:
                break
                
            title = path['title']
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            # Get full metadata
            metadata = self.get_full_metadata(title)
            
            passages.append({
                'title': title,
                'metadata': metadata,
                'matched_path': path['key_path'],
                'matched_value': path['value'],
                'score': path['score']
            })
        
        return passages
    
    def _build_simple_previous_context(self, current_sq: SubQuestion, decomposition: QueryDecomposition) -> str:
        """
        Build simple previous context with only SQ answers (no passages).
        Only includes direct dependencies' answers.
        """
        if not current_sq.depends_on:
            return ""
        
        context_parts = []
        for dep_id in current_sq.depends_on:
            dep_sq = decomposition.get_subquestion(dep_id)
            if dep_sq and dep_sq.answer:
                context_parts.append(f"{dep_sq.id}: {dep_sq.question}")
                context_parts.append(f"Answer: {dep_sq.answer}")
                context_parts.append("")
        
        if context_parts:
            return "Previous Sub-Questions:\n" + "\n".join(context_parts)
        return ""
    
    def _collect_all_unique_passages(self, decomposition: QueryDecomposition) -> List[Dict]:
        """
        Collect all unique passages from all sub-questions.
        Returns deduplicated list by title.
        """
        seen_titles = set()
        unique_passages = []
        
        for sq in decomposition.subquestions:
            if hasattr(sq, 'retrieved_passages') and sq.retrieved_passages:
                for passage in sq.retrieved_passages:
                    title = passage.get('title', '')
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        unique_passages.append(passage)
        
        return unique_passages
    
    async def generate_final_answer(
        self,
        main_query: str,
        decomposition: QueryDecomposition,
        all_passages: List[Dict]
    ) -> str:
        """
        Generate final answer for main query using all collected passages.
        """
        # Build sub-question chain
        chain_parts = []
        for sq in decomposition.subquestions:
            chain_parts.append(f"{sq.id}: {sq.question}")
            chain_parts.append(f"Answer: {sq.answer if sq.answer else '(Not answered)'}")
            chain_parts.append("")
        subquestion_chain = '\n'.join(chain_parts)
        
        # Format passages
        passage_texts = []
        for i, p in enumerate(all_passages, 1):
            parts = [f"[{i}] {p['title']}"]
            
            if p.get('metadata'):
                metadata = p['metadata']
                excluded_keys = {'title', 'type', 'subtype'}
                for key, value in metadata.items():
                    if key not in excluded_keys and value:
                        value_str = str(value) if not isinstance(value, (dict, list)) else json.dumps(value, ensure_ascii=False)
                        parts.append(f"  {key}: {value_str}")
            
            passage_texts.append('\n'.join(parts))
        
        passages_text = '\n\n'.join(passage_texts) if passage_texts else "No passages."
        
        # Use FINAL_ANSWER_SYNTHESIS_PROMPT
        prompt = FINAL_ANSWER_SYNTHESIS_PROMPT.replace("{{main_question}}", main_query)
        prompt = prompt.replace("{{subquestion_chain}}", subquestion_chain)
        prompt = prompt.replace("{{passages}}", passages_text)
        
        response = await self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise question answering system. Give short, direct answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        answer = response.choices[0].message.content.strip()
        
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        
        return answer
    
    async def answer_subquestion(
        self,
        sq: SubQuestion,
        decomposition: QueryDecomposition,
        is_final_sq: bool = False
    ) -> Dict:
        """
        Answer a single sub-question.
        
        Args:
            sq: SubQuestion to answer
            decomposition: Full decomposition for context
            is_final_sq: Whether this is the final sub-question
            
        Returns:
            Dict with success, answer, passages, etc.
        """
        try:
            # Substitute placeholders
            actual_question = substitute_answers(sq.question, decomposition.subquestions)
            
            # Build simple previous context (only answers, no passages)
            previous_context = self._build_simple_previous_context(sq, decomposition)
            
            # Retrieve passages
            passages = await self.retrieve_for_query(actual_question)
            
            if self.verbose:
                print(f"\n   Retrieved {len(passages)} passages:")
                for p in passages:
                    print(f"     - {p['title']} (path: {p['matched_path']}, score: {p['score']:.3f})")
            
            # Generate answer (using is_final_sq)
            answer = await self.generate_answer(
                actual_question,
                passages,
                previous_context,
                decomposition.main_query,
                is_final_sq=is_final_sq
            )
            
            # Update SubQuestion
            sq.answer = answer
            sq.retrieved_passages = passages
            
            return {
                'success': True,
                'answer': answer,
                'actual_question': actual_question,
                'passages': passages
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def generate_answer(
        self,
        question: str,
        passages: List[Dict],
        previous_context: str,
        main_query: str,
        is_final_sq: bool = False
    ) -> str:
        """
        Generate answer from passages using LLM.
        Uses the same prompts as sequential_answering.py.
        """
        
        # Format passages (same format as sequential_answering.py)
        passage_texts = []
        for i, p in enumerate(passages, 1):
            parts = [f"[{i}] {p['title']}"]
            
            # Include full metadata (excluding type/subtype)
            if p['metadata']:
                metadata = p['metadata']
                excluded_keys = {'title', 'type', 'subtype'}
                for key, value in metadata.items():
                    if key not in excluded_keys and value:
                        value_str = str(value) if not isinstance(value, (dict, list)) else json.dumps(value, ensure_ascii=False)
                        parts.append(f"  {key}: {value_str}")
            
            passage_texts.append('\n'.join(parts))
        
        passages_text = '\n\n'.join(passage_texts) if passage_texts else "No passages retrieved."
        
        # Choose prompt based on whether this is the final SQ (same as sequential_answering.py)
        if is_final_sq:
            prompt = FINAL_SUBQUESTION_ANSWERING_PROMPT.replace(
                "{{subquestion}}", question
            )
        else:
            prompt = DETAILED_SUBQUESTION_ANSWERING_PROMPT.replace(
                "{{subquestion}}", question
            )
        
        prompt = prompt.replace("{{passages}}", passages_text)
        prompt = prompt.replace(
            "{{previous_context}}", 
            previous_context if previous_context else "(None)"
        )
        prompt = prompt.replace(
            "{{main_query}}", 
            main_query if main_query else "(No main query provided)"
        )
        
        # LLM call
        max_tokens = 100 if is_final_sq else 200
        
        response = await self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise question answering system. Give short, direct answers."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Clean up answer
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        
        return answer
    
    async def process_question(self, question: str) -> Dict:
        """
        Process a single question through the full pipeline.
        
        Args:
            question: The main question
            
        Returns:
            Dict with results
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Processing: {question}")
            print(f"{'='*80}")
        
        # Step 1: Query Decomposition
        if self.verbose:
            print(f"\n[Step 1] Query Decomposition...")
        
        decomp_result = await decompose_query(self.client, question)
        
        if not decomp_result['success']:
            return {
                'success': False,
                'error': f"Decomposition failed: {decomp_result.get('error')}"
            }
        
        decomposition = decomp_result['decomposition']
        
        if self.verbose:
            print(f"   Type: {decomposition.question_type}")
            print(f"   Sub-questions: {len(decomposition.subquestions)}")
            for sq in decomposition.subquestions:
                deps = f" [depends: {', '.join(sq.depends_on)}]" if sq.depends_on else ""
                print(f"     {sq.id}: {sq.question}{deps}")
        
        # Step 2: Answer sub-questions
        if self.verbose:
            print(f"\n[Step 2] Answering Sub-Questions...")
        
        batches = get_execution_order(decomposition)
        total_batches = len(batches)
        
        for batch_idx, batch in enumerate(batches, 1):
            is_final_batch = (batch_idx == total_batches)
            
            if self.verbose:
                print(f"\n   Batch {batch_idx}/{total_batches}: {batch}")
            
            if len(batch) == 1:
                # Sequential
                sq_id = batch[0]
                sq = decomposition.get_subquestion(sq_id)
                is_final_sq = is_final_batch  # Last batch = final SQ
                
                if self.verbose:
                    sq_label = " [FINAL]" if is_final_sq else ""
                    print(f"\n   {sq.id}: {sq.question}{sq_label}")
                
                result = await self.answer_subquestion(sq, decomposition, is_final_sq=is_final_sq)
                
                if self.verbose:
                    if result['success']:
                        print(f"   ✓ Answer: {result['answer']}")
                    else:
                        print(f"   ✗ Error: {result.get('error')}")
            else:
                # Parallel
                tasks = []
                for sq_id in batch:
                    sq = decomposition.get_subquestion(sq_id)
                    is_final_sq = is_final_batch
                    tasks.append(self.answer_subquestion(sq, decomposition, is_final_sq=is_final_sq))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for sq_id, result in zip(batch, results):
                    sq = decomposition.get_subquestion(sq_id)
                    if self.verbose:
                        print(f"\n   {sq.id}: {sq.question}")
                        if isinstance(result, Exception):
                            print(f"   ✗ Error: {result}")
                        elif result['success']:
                            print(f"   ✓ Answer: {result['answer']}")
                        else:
                            print(f"   ✗ Error: {result.get('error')}")
        
        # Step 3: Final Answer - Answer Main Query with all unique passages
        all_passages = self._collect_all_unique_passages(decomposition)
        
        if self.verbose:
            print(f"\n[Step 3] Final Answer (Main Query with {len(all_passages)} unique passages)")
        
        final_answer = await self.generate_final_answer(
            main_query=question,
            decomposition=decomposition,
            all_passages=all_passages
        )
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"FINAL ANSWER: {final_answer}")
            print(f"Time: {total_time:.2f}s | Passages used: {len(all_passages)}")
            print(f"{'='*80}")
        
        return {
            'success': True,
            'question': question,
            'final_answer': final_answer,
            'decomposition': decomposition.to_dict(),
            'time': total_time,
            'num_passages': len(all_passages)
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()


async def test_pipeline():
    """Test the new pipeline with HotpotQA questions."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("="*80)
    print("Testing New Multi-hop Pipeline v2")
    print("="*80)
    
    # Initialize components
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6
    )
    
    pipeline = NewMultihopPipeline(
        client=client,
        retriever=retriever,
        top_k=3,
        verbose=True
    )
    
    # Load HotpotQA sample questions
    with open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8') as f:
        hotpot_data = json.load(f)
    
    # Test with first 3 questions
    test_questions = [
        {"question": hotpot_data[0]["question"], "answer": hotpot_data[0]["answer"]},
        {"question": hotpot_data[1]["question"], "answer": hotpot_data[1]["answer"]},
        {"question": hotpot_data[2]["question"], "answer": hotpot_data[2]["answer"]},
    ]
    
    print(f"\nTesting with {len(test_questions)} HotpotQA questions\n")
    
    for i, q in enumerate(test_questions, 1):
        print(f"\n{'#'*80}")
        print(f"Question {i}: {q['question']}")
        print(f"Gold Answer: {q['answer']}")
        print(f"{'#'*80}")
        
        result = await pipeline.process_question(q['question'])
        
        if result['success']:
            print(f"\n>>> Predicted Answer: {result['final_answer']}")
            print(f">>> Gold Answer: {q['answer']}")
            print(f">>> Time: {result['time']:.2f}s")
        else:
            print(f">>> Error: {result.get('error')}")
    
    pipeline.close()


if __name__ == "__main__":
    asyncio.run(test_pipeline())
