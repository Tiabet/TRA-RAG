#!/usr/bin/env python3
"""
New Multi-hop Pipeline v3 - No Decomposition (Ablation Study)
==============================================================
Same as v3 but WITHOUT query decomposition.
Directly retrieves passages using the original question and generates answer.

Purpose: Ablation study to measure the impact of query decomposition.

Pipeline:
1. Direct Retrieval: Original query → Hybrid Search → Top-k passages
2. Direct Answer Generation: LLM generates answer from retrieved passages
"""

import asyncio
import json
import time
import sqlite3
from typing import Dict, List, Optional
from openai import AsyncOpenAI

from hybrid_path_retriever import HybridPathRetriever
from llm_logger import log_llm_call


# Direct answering prompt (no decomposition context)
DIRECT_ANSWER_PROMPT = """---Role---
You are a multi-hop retrieval-augmented assistant.

---Goal---
Read the provided Information and generate the correct answer to the Question.
Use ONLY the given Information to derive your answer.

---Critical Instructions---
1. Read ALL provided passages carefully
2. Extract relevant facts from the passages
3. Combine information from multiple passages if needed
4. Perform simple reasoning if required (arithmetic, temporal logic, comparisons)

---Target response length and format---
- One-word or minimal-phrase answer (max 5 words).

---Response Rules---
✓ Use ONLY the information provided in the passages
✓ Check ALL passages thoroughly
✓ You CAN perform simple reasoning on passage information
✓ Answer must be short and concise
✓ Answer language must match the Question language
✗ Do NOT use external knowledge not present in passages
✗ Do NOT hallucinate or invent facts
✗ ONLY respond "Insufficient information." if passages truly lack the needed information

---Information---
{passages}

---Question---
{question}

---Answer---
Provide only the answer (max 5 words).
"""


class NewMultihopPipelineV3NoDecomp:
    """Pipeline using hybrid retrieval + original passages, WITHOUT query decomposition."""
    
    def __init__(
        self,
        client: AsyncOpenAI,
        retriever: HybridPathRetriever,
        hotpotqa_path: str = 'HotpotQA/hotpotqa_sample_200.json',
        db_path: str = 'HotpotQA/metadata_v3.db',
        top_k: int = 5,  # More passages since no decomposition
        verbose: bool = True
    ):
        self.client = client
        self.retriever = retriever
        self.db_path = db_path
        self.top_k = top_k
        self.verbose = verbose
        
        # Load original passages indexed by title
        self.original_passages = self._load_original_passages(hotpotqa_path)
        if self.verbose:
            print(f"✓ Loaded {len(self.original_passages)} original passages")
        
        # Connect to database
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def _load_original_passages(self, hotpotqa_path: str) -> Dict[str, str]:
        """Load original passages from dataset and index by title."""
        with open(hotpotqa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        passages = {}
        for item in data:
            for title, sentences in item.get('context', []):
                if title not in passages:
                    full_text = ''.join(sentences).strip()
                    passages[title] = full_text
        
        return passages
    
    def get_original_passage(self, title: str) -> Optional[str]:
        """Get original passage text for a title."""
        return self.original_passages.get(title)
    
    async def retrieve_for_query(self, query: str) -> List[Dict]:
        """Retrieve passages for a query using hybrid search."""
        fetch_k = self.top_k * 10
        paths = await self.retriever.search_hybrid(query, top_k=fetch_k)
        
        seen_titles = set()
        passages = []
        
        for path in paths:
            if len(passages) >= self.top_k:
                break
                
            title = path['title']
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            original_passage = self.get_original_passage(title)
            
            passages.append({
                'title': title,
                'original_passage': original_passage,
                'score': path['score']
            })
        
        return passages
    
    async def generate_answer(self, question: str, passages: List[Dict]) -> str:
        """Generate answer directly from retrieved passages."""
        passage_texts = []
        for i, p in enumerate(passages, 1):
            title = p['title']
            original_text = p.get('original_passage', '')
            
            if original_text:
                passage_texts.append(f"[{i}] {title}\n{original_text}")
            else:
                passage_texts.append(f"[{i}] {title}\n(No content available)")
        
        passages_text = '\n\n'.join(passage_texts) if passage_texts else "No passages retrieved."
        
        prompt = DIRECT_ANSWER_PROMPT.format(
            passages=passages_text,
            question=question
        )
        
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
        
        log_llm_call(
            call_type="Direct Answer (V3-NoDecomp)",
            input_text=prompt,
            output_text=answer,
            context={"question": question, "num_passages": len(passages)}
        )
        
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        
        return answer
    
    async def process_question(self, question: str) -> Dict:
        """Process a question WITHOUT decomposition."""
        start_time = time.time()
        
        try:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Question: {question}")
                print(f"{'='*60}")
                print("\n[1] Direct retrieval (no decomposition)...")
            
            # Step 1: Direct retrieval
            passages = await self.retrieve_for_query(question)
            
            if self.verbose:
                print(f"   Retrieved {len(passages)} passages:")
                for p in passages:
                    has_original = "✓" if p.get('original_passage') else "✗"
                    print(f"     - {p['title']} (original: {has_original}, score: {p['score']:.3f})")
            
            # Step 2: Generate answer directly
            if self.verbose:
                print(f"\n[2] Generating answer directly...")
            
            final_answer = await self.generate_answer(question, passages)
            
            elapsed = time.time() - start_time
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Final Answer: {final_answer}")
                print(f"Time: {elapsed:.2f}s | Passages: {len(passages)}")
                print(f"{'='*60}")
            
            return {
                'success': True,
                'final_answer': final_answer,
                'decomposition': None,
                'num_passages': len(passages),
                'retrieved_passages': passages,
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
