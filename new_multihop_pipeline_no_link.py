#!/usr/bin/env python3
"""
New Multi-hop Pipeline v3
==========================
Same as v2 but uses original passages for answer generation instead of metadata.

Key difference from v2:
- Retrieval: Uses metadata-based hybrid search (same as v2)
- Answer Generation: Uses original passages from HotpotQA context

Pipeline:
1. Query Decomposition (existing)
2. For each SQ:
   - Hybrid Search (BM25 + Dense) on metadata paths → Top-k titles
   - Title → Original Passage mapping
   - SQ Answering (original passage-based)
3. Final Answer: Main Query 답변 (모든 SQ의 unique original passages 사용)
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

# Import prompts from Prompt folder
from Prompt.answer_prompt import (
    DETAILED_SUBQUESTION_ANSWERING_PROMPT,
    FINAL_SUBQUESTION_ANSWERING_PROMPT,
    FINAL_ANSWER_SYNTHESIS_PROMPT,
)

# Import logger
from llm_logger import log_llm_call, log_llm_error


class NewMultihopPipelineV3:
    """Pipeline using hybrid retrieval + original passages for answering."""
    
    def __init__(
        self,
        client: AsyncOpenAI,
        retriever: HybridPathRetriever,
        hotpotqa_path: str = 'HotpotQA/hotpotqa_sample_200.json',
        db_path: str = 'HotpotQA/metadata_v4aligned.db',
        top_k: int = 3,
        verbose: bool = True
    ):
        """
        Args:
            client: AsyncOpenAI client for LLM calls
            retriever: HybridPathRetriever instance
            hotpotqa_path: Path to original HotpotQA data (for original passages)
            db_path: Path to metadata database (for metadata lookup, optional)
            top_k: Number of passages to retrieve per query
            verbose: Print progress
        """
        self.client = client
        self.retriever = retriever
        self.db_path = db_path
        self.top_k = top_k
        self.verbose = verbose
        
        # Load original passages indexed by title
        self.original_passages = self._load_original_passages(hotpotqa_path)
        if self.verbose:
            print(f"[OK] Loaded {len(self.original_passages)} original passages")
        
        # Connect to database for metadata lookup (optional, for debugging)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def _load_original_passages(self, hotpotqa_path: str) -> Dict[str, str]:
        """
        Load original passages from HotpotQA and index by title.
        
        Returns:
            Dict mapping title -> concatenated passage text
        """
        with open(hotpotqa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        passages = {}
        for item in data:
            for title, sentences in item.get('context', []):
                if title not in passages:
                    # Concatenate sentences into full passage
                    full_text = ''.join(sentences).strip()
                    passages[title] = full_text
        
        return passages
    
    def get_original_passage(self, title: str) -> Optional[str]:
        """Get original passage text for a title."""
        return self.original_passages.get(title)
    
    def get_full_metadata(self, title: str, doc_id: Optional[str] = None) -> Optional[Dict]:
        """Get full metadata from database (doc_id-aware, legacy fallback)."""
        cursor = self.conn.cursor()

        if doc_id:
            try:
                cursor.execute(
                    "SELECT metadata_json FROM metadata WHERE doc_id = ?",
                    (doc_id,)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row['metadata_json'])
            except Exception:
                # Legacy schema fallback below
                pass

        try:
            cursor.execute(
                "SELECT metadata_json FROM metadata WHERE title = ?",
                (title,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row['metadata_json'])
        except Exception:
            return None

        return None
    
    async def retrieve_for_query(self, query: str) -> List[Dict]:
        """
        Retrieve passages for a query using hybrid search.
        Returns top_k unique titles with their original passages.
        
        Args:
            query: The query text
            
        Returns:
            List of passage dicts with title, original_passage, and metadata
        """
        # Request more paths to ensure we get enough unique titles
        fetch_k = self.top_k * 10
        paths = await self.retriever.search_hybrid(query, top_k=fetch_k)
        
        # Get unique source titles until we have top_k
        seen_source_titles = set()
        passages = []
        
        for path in paths:
            if len(passages) >= self.top_k:
                break
                
            entity_title = path['title']
            source_title = path.get('source_title') or entity_title
            if source_title in seen_source_titles:
                continue
            seen_source_titles.add(source_title)
            
            # Get original passage
            original_passage = self.get_original_passage(source_title)
            
            # Get metadata (for reference)
            metadata = self.get_full_metadata(entity_title, doc_id=path.get('doc_id'))
            
            passages.append({
                'title': source_title,
                'source_title': source_title,
                'entity_title': entity_title,
                'doc_id': path.get('doc_id'),
                'original_passage': original_passage,
                'metadata': metadata,
                'matched_path': path['key_path'],
                'matched_value': path['value'],
                'score': path['score'],
                'bm25_score': path.get('bm25_score', 0),
                'dense_score': path.get('dense_score', 0)
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
                context_parts.append(f"{dep_id}: {dep_sq.question}")
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
        Generate final answer for main query using all collected original passages.
        """
        # Build sub-question chain
        chain_parts = []
        for sq in decomposition.subquestions:
            chain_parts.append(f"{sq.id}: {sq.question}")
            chain_parts.append(f"Answer: {sq.answer if sq.answer else '(Not answered)'}")
            chain_parts.append("")
        subquestion_chain = '\n'.join(chain_parts)
        
        # Format passages using ORIGINAL passages
        passage_texts = []
        for i, p in enumerate(all_passages, 1):
            title = p['title']
            original_text = p.get('original_passage', '')
            
            if original_text:
                passage_texts.append(f"[{i}] {title}\n{original_text}")
            else:
                # Fallback to metadata if original not available
                passage_texts.append(f"[{i}] {title}\n(No original passage available)")
        
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
            temperature=0.0,
            max_tokens=100
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Log the LLM call
        log_llm_call(
            call_type="Final Answer Synthesis (V3-Original)",
            input_text=prompt,
            output_text=answer,
            context={"main_query": main_query, "num_passages": len(all_passages)}
        )
        
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
        Answer a single sub-question using original passages.
        """
        try:
            # Substitute placeholders
            actual_question = substitute_answers(sq.question, decomposition.subquestions)
            
            # Build simple previous context (only answers, no passages)
            previous_context = self._build_simple_previous_context(sq, decomposition)
            
            # Retrieve passages (with original text)
            passages = await self.retrieve_for_query(actual_question)
            
            if self.verbose:
                print(f"\n   Retrieved {len(passages)} passages:")
                for p in passages:
                    has_original = "✓" if p.get('original_passage') else "✗"
                    print(f"     - {p['title']} (original: {has_original}, score: {p['score']:.3f})")
            
            # Generate answer using original passages
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
        Generate answer from ORIGINAL passages using LLM.
        """
        
        # Format passages using ORIGINAL text
        passage_texts = []
        for i, p in enumerate(passages, 1):
            title = p['title']
            original_text = p.get('original_passage', '')
            
            if original_text:
                passage_texts.append(f"[{i}] {title}\n{original_text}")
            else:
                # Fallback to metadata if no original passage
                if p.get('metadata'):
                    metadata = p['metadata']
                    parts = [f"[{i}] {title}"]
                    excluded_keys = {'title'}
                    for key, value in metadata.items():
                        if key not in excluded_keys and value:
                            value_str = str(value) if not isinstance(value, (dict, list)) else json.dumps(value, ensure_ascii=False)
                            parts.append(f"  {key}: {value_str}")
                    passage_texts.append('\n'.join(parts))
                else:
                    passage_texts.append(f"[{i}] {title}\n(No content available)")
        
        passages_text = '\n\n'.join(passage_texts) if passage_texts else "No passages retrieved."
        
        # Choose prompt based on is_final_sq
        if is_final_sq:
            prompt_template = FINAL_SUBQUESTION_ANSWERING_PROMPT
        else:
            prompt_template = DETAILED_SUBQUESTION_ANSWERING_PROMPT
        
        # Fill in the prompt
        prompt = prompt_template.replace("{{subquestion}}", question)
        prompt = prompt.replace("{{passages}}", passages_text)
        prompt = prompt.replace("{{previous_context}}", previous_context if previous_context else "None")
        prompt = prompt.replace("{{main_query}}", main_query)
        
        response = await self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise question answering system."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=150
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Log the LLM call
        log_llm_call(
            call_type="Subquestion Answering (V3-Original)",
            input_text=prompt,
            output_text=answer,
            context={
                "question": question,
                "main_query": main_query,
                "is_final_sq": is_final_sq,
                "num_passages": len(passages)
            }
        )
        
        # Clean up answer format
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        
        return answer
    
    async def process_question(self, question: str) -> Dict:
        """
        Process a multi-hop question end-to-end using original passages.
        """
        start_time = time.time()
        
        try:
            # Step 1: Decompose query
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Question: {question}")
                print(f"{'='*60}")
                print("\n[1] Decomposing query...")
            
            decomp_result = await decompose_query(self.client, question)
            
            if not decomp_result['success']:
                return {
                    'success': False,
                    'error': f"Decomposition failed: {decomp_result.get('error')}",
                    'time': time.time() - start_time
                }
            
            decomposition = decomp_result['decomposition']
            
            if self.verbose:
                print(f"   Main query: {decomposition.main_query}")
                print(f"   Sub-questions: {len(decomposition.subquestions)}")
                for sq in decomposition.subquestions:
                    deps = f" (depends: {sq.depends_on})" if sq.depends_on else ""
                    print(f"     {sq.id}: {sq.question}{deps}")
            
            # Step 2: Answer sub-questions in order
            batches = get_execution_order(decomposition)
            
            if self.verbose:
                print(f"\n[2] Answering sub-questions in {len(batches)} batches...")
            
            total_batches = len(batches)
            for batch_idx, batch in enumerate(batches, 1):
                is_final_batch = (batch_idx == total_batches)
                
                for sq_id in batch:
                    sq = decomposition.get_subquestion(sq_id)
                    is_final = is_final_batch and (sq_id == batch[-1])
                
                    if self.verbose:
                        print(f"\n   --- {sq_id} ---")
                        print(f"   Q: {sq.question}")
                
                    result = await self.answer_subquestion(sq, decomposition, is_final_sq=is_final)
                
                    if result['success']:
                        if self.verbose:
                            print(f"   A: {result['answer']}")
                    else:
                        if self.verbose:
                            print(f"   Error: {result['error']}")
            
            # Step 3: Generate final answer
            if self.verbose:
                print(f"\n[3] Generating final answer...")
            
            all_passages = self._collect_all_unique_passages(decomposition)
            final_answer = await self.generate_final_answer(
                decomposition.main_query,
                decomposition,
                all_passages
            )
            
            elapsed = time.time() - start_time
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Final Answer: {final_answer}")
                print(f"Time: {elapsed:.2f}s | Passages: {len(all_passages)}")
                print(f"{'='*60}")
            
            return {
                'success': True,
                'final_answer': final_answer,
                'decomposition': {
                    'main_query': decomposition.main_query,
                    'subquestions': [
                        {
                            'id': sq.id,
                            'question': sq.question,
                            'answer': sq.answer,
                            'depends_on': sq.depends_on,
                            'retrieved_passages': getattr(sq, 'retrieved_passages', [])
                        }
                        for sq in decomposition.subquestions
                    ]
                },
                'num_passages': len(all_passages),
                'time': elapsed
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            if self.verbose:
                print(f"\nError: {e}")
            return {
                'success': False,
                'error': str(e),
                'time': elapsed
            }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# Test function
async def test_pipeline():
    """Quick test of the v3 pipeline."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    print("="*80)
    print("Testing New Multi-hop Pipeline v3 (Original Passages)")
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
    
    pipeline = NewMultihopPipelineV3(
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
    from llm_logger import init_logger, finalize_log

    init_logger()
    try:
        asyncio.run(test_pipeline())
    finally:
        finalize_log()
