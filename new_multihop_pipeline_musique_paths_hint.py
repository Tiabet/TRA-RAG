#!/usr/bin/env python3
"""\
MuSiQue Multi-hop Pipeline (Paths-as-Hints + rag_qa_cot one-shot)
===============================================================

Based on `new_multihop_pipeline_paths_hint.py` (v11 paths-as-hints), but:
- Defaults to MuSiQue dataset files.
- Applies `Prompt/rag_qa_cot.py` one-shot chat prompt template to BOTH:
    - sub-question answering
    - final main-query answering
    while still leveraging the repo's existing text prompts
    (`DETAILED_SUBQUESTION_ANSWERING_PROMPT`, `FINAL_SUBQUESTION_ANSWERING_PROMPT`,
     `FINAL_ANSWER_SYNTHESIS_PROMPT`).

Output remains compatible with:
- evaluate_retrieval.py (reads retrieved_passages titles)
- evaluate_mrqa.py (reads final_answer/predicted_answer)
- llm_evaluation.py (reads predicted_answer)

All LLM calls are logged via llm_logger.
"""

import asyncio
import copy
import json
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from query_decomposition import (
    QueryDecomposition,
    SubQuestion,
    decompose_query,
    get_execution_order,
    substitute_answers,
)
from hybrid_path_retriever import HybridPathRetriever

from Prompt.answer import (
    DETAILED_SUBQUESTION_ANSWERING_PROMPT,
    FINAL_SUBQUESTION_ANSWERING_PROMPT,
)
from Prompt.subquestion_answering_prompt import FINAL_ANSWER_SYNTHESIS_PROMPT
from Prompt.rag_qa_cot import prompt_template as RAG_QA_COT_PROMPT_TEMPLATE

from llm_logger import log_llm_call


class MuSiQueMultihopPipelineV11PathsHint:
    """MuSiQue-dedicated pipeline using hybrid retrieval + original passages + top path hints."""

    def __init__(
        self,
        client: AsyncOpenAI,
        retriever: HybridPathRetriever,
        musique_path: str = 'MuSiQue/musique_sample_200.json',
        db_path: str = 'MuSiQue/metadata_v3.db',
        top_k_passages: int = 3,
        top_k_paths: int = 10,
        path_fetch_k: int = 50,
        verbose: bool = True,
        log_messages: bool = False,
    ):
        self.client = client
        self.retriever = retriever
        self.db_path = db_path
        self.top_k_passages = top_k_passages
        self.top_k_paths = top_k_paths
        self.path_fetch_k = max(path_fetch_k, top_k_paths, top_k_passages)
        self.verbose = verbose
        self.log_messages = log_messages

        self.original_passages = self._load_original_passages(musique_path)
        if self.verbose:
            print(f"[OK] Loaded {len(self.original_passages)} original passages")

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def _load_original_passages(self, musique_path: str) -> Dict[str, str]:
        with open(musique_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        passages: Dict[str, str] = {}
        for item in data:
            for title, sentences in item.get('context', []):
                if title not in passages:
                    full_text = ''.join(sentences).strip()
                    passages[title] = full_text
        return passages

    def get_original_passage(self, title: str) -> Optional[str]:
        return self.original_passages.get(title)

    def get_full_metadata(self, title: str, doc_id: Optional[str] = None) -> Optional[Dict]:
        cursor = self.conn.cursor()

        if doc_id:
            try:
                cursor.execute(
                    "SELECT metadata_json FROM metadata WHERE doc_id = ?",
                    (doc_id,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row['metadata_json'])
            except Exception:
                pass

        try:
            cursor.execute(
                "SELECT metadata_json FROM metadata WHERE title = ?",
                (title,),
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row['metadata_json'])
        except Exception:
            return None

        return None

    async def retrieve_for_query(self, query: str) -> Tuple[List[Dict], List[Dict]]:
        """Return (top_unique_passages, top_paths_as_hints)."""
        fetched_paths = await self.retriever.search_hybrid(
            query,
            top_k=self.path_fetch_k,
            bm25_candidates=max(50, self.path_fetch_k),
            dense_candidates=max(50, self.path_fetch_k),
        )

        top_paths = fetched_paths[: self.top_k_paths]

        seen_source_titles = set()
        passages: List[Dict] = []

        for path in fetched_paths:
            if len(passages) >= self.top_k_passages:
                break

            entity_title = path.get('entity_title') or path.get('title')
            source_title = path.get('source_title') or entity_title
            if source_title in seen_source_titles:
                continue
            seen_source_titles.add(source_title)

            original_passage = self.get_original_passage(source_title)
            metadata = self.get_full_metadata(str(entity_title), doc_id=path.get('doc_id'))

            passages.append({
                'title': source_title,
                'source_title': source_title,
                'entity_title': str(entity_title),
                'doc_id': path.get('doc_id'),
                'original_passage': original_passage,
                'metadata': metadata,
                'matched_path': path.get('key_path'),
                'matched_value': path.get('value'),
                'score': path.get('score'),
                'bm25_score': path.get('bm25_score', 0),
                'dense_score': path.get('dense_score', 0),
            })

        return passages, top_paths

    async def retrieve_paths_for_query(self, query: str) -> List[Dict]:
        """Return top-k paths-as-hints for a query.

        Used for final answering (main query), where passages come from all sub-questions.
        """
        fetched_paths = await self.retriever.search_hybrid(
            query,
            top_k=self.path_fetch_k,
            bm25_candidates=max(50, self.path_fetch_k),
            dense_candidates=max(50, self.path_fetch_k),
        )
        return fetched_paths[: self.top_k_paths]

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

    def _collect_all_unique_passages(self, decomposition: QueryDecomposition) -> List[Dict]:
        seen_titles = set()
        unique_passages: List[Dict] = []

        for sq in decomposition.subquestions:
            if hasattr(sq, 'retrieved_passages') and sq.retrieved_passages:
                for passage in sq.retrieved_passages:
                    title = passage.get('title', '')
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        unique_passages.append(passage)

        return unique_passages

    @staticmethod
    def _format_paths_as_hints(paths: List[Dict]) -> str:
        if not paths:
            return "(No paths.)"

        lines = []
        for i, p in enumerate(paths, 1):
            source_title = p.get('source_title') or p.get('title') or ''
            entity_title = p.get('entity_title') or p.get('title') or ''
            key_path = p.get('key_path', '')
            value = p.get('value', '')
            if isinstance(value, str) and len(value) > 220:
                value = value[:220] + "..."
            lines.append(
                f"[{i}] source_title: {source_title} | entity_title: {entity_title}\n"
                f"  {key_path}: {value}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_passages_original(passages: List[Dict]) -> str:
        if not passages:
            return "No passages retrieved."

        passage_texts = []
        for i, p in enumerate(passages, 1):
            title = p.get('title', '')
            original_text = p.get('original_passage', '')

            if original_text:
                passage_texts.append(f"[{i}] {title}\n{original_text}")
            else:
                if p.get('metadata'):
                    metadata = p['metadata']
                    parts = [f"[{i}] {title}"]
                    excluded_keys = {'title'}
                    for key, value in metadata.items():
                        if key in excluded_keys or not value:
                            continue
                        value_str = (
                            str(value)
                            if not isinstance(value, (dict, list))
                            else json.dumps(value, ensure_ascii=False)
                        )
                        parts.append(f"  {key}: {value_str}")
                    passage_texts.append("\n".join(parts))
                else:
                    passage_texts.append(f"[{i}] {title}\n(No content available)")

        return "\n\n".join(passage_texts)

    async def generate_answer(
        self,
        question: str,
        passages: List[Dict],
        top_paths: List[Dict],
        previous_context: str,
        main_query: str,
        is_final_sq: bool = False,
    ) -> str:
        passages_text = self._format_passages_original(passages)
        paths_text = self._format_paths_as_hints(top_paths)

        combined_info = (
            "---Top Retrieved Metadata Paths (STRONG HINTS)---\n"
            "The paths below are strong hints for where the answer might be found. "
            "Use them to focus your reading of the passages, but do NOT treat them as guaranteed truth.\n\n"
            f"{paths_text}\n\n"
            "---Original Passages (Top 3)---\n"
            f"{passages_text}"
        )

        prev = previous_context if previous_context else "None"
        information = f"---Previous Context---\n{prev}\n\n{combined_info}"
        user_content = self._rag_qa_cot_user_content(information=information, query=question)

        messages = self._build_rag_qa_cot_messages(user_content)
        response = await self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=150,
        )

        answer = response.choices[0].message.content.strip()

        context = {
            "question": question,
            "main_query": main_query,
            "is_final_sq": is_final_sq,
            "num_passages": len(passages),
            "num_paths": len(top_paths),
        }
        if self.log_messages:
            context["chat_messages"] = self._messages_for_log(messages)

        log_llm_call(
            call_type="Subquestion Answering (MuSiQue-V11-PathsHint + rag_qa_cot)",
            input_text=user_content,
            output_text=answer,
            context=context,
        )

        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        return answer

    def _build_rag_qa_cot_messages(self, prompt_user_text: str) -> List[Dict[str, str]]:
        """Render chat messages from `Prompt/rag_qa_cot.py` prompt_template.

        We send the full one-shot template in a single request, and only replace
        the final user placeholder (${prompt_user}).
        """
        messages = copy.deepcopy(RAG_QA_COT_PROMPT_TEMPLATE)
        rendered = False
        for msg in messages:
            if msg.get('role') == 'user' and msg.get('content') == '${prompt_user}':
                msg['content'] = prompt_user_text
                rendered = True
                break
        if not rendered:
            messages.append({"role": "user", "content": prompt_user_text})
        return messages

    @staticmethod
    def _messages_for_log(messages: List[Dict[str, str]]) -> str:
        """Pretty JSON string for logging the exact chat payload."""
        try:
            return json.dumps(messages, ensure_ascii=False, indent=2)
        except Exception:
            return str(messages)

    @staticmethod
    def _rag_qa_cot_user_content(information: str, query: str) -> str:
        """Build user message content matching rag_qa_cot one-shot format."""
        info = information.strip() if information else ""
        q = query.strip() if query else ""
        return f"---Information---\n{info}\n\n---Query---\n{q}"

    async def generate_final_answer(
        self,
        main_query: str,
        decomposition: QueryDecomposition,
        all_passages: List[Dict],
    ) -> str:
        chain_parts = []
        for sq in decomposition.subquestions:
            chain_parts.append(f"{sq.id}: {sq.question}")
            chain_parts.append(f"Answer: {sq.answer if sq.answer else '(Not answered)'}")
            chain_parts.append("")
        subquestion_chain = '\n'.join(chain_parts)

        passages_text = self._format_passages_original(all_passages)

        # Re-retrieve path hints for the MAIN query (so final answering also sees paths).
        top_paths = await self.retrieve_paths_for_query(main_query)
        paths_text = self._format_paths_as_hints(top_paths)

        combined_info = (
            "---Top Retrieved Metadata Paths (STRONG HINTS)---\n"
            "The paths below are strong hints for where the answer might be found. "
            "Use them to focus your reading of the passages, but do NOT treat them as guaranteed truth.\n\n"
            f"{paths_text}\n\n"
            "---All Retrieved Passages---\n"
            f"{passages_text}"
        )

        # Match rag_qa_cot one-shot format for the REAL user message.
        information = (
            f"---Sub-Question Chain---\n{subquestion_chain}\n\n"
            f"{combined_info}"
        )
        user_content = self._rag_qa_cot_user_content(information=information, query=main_query)

        messages = self._build_rag_qa_cot_messages(user_content)

        response = await self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=100,
        )

        answer = response.choices[0].message.content.strip()

        context = {
            "main_query": main_query,
            "num_passages": len(all_passages),
            "num_paths": len(top_paths),
        }
        if self.log_messages:
            context["chat_messages"] = self._messages_for_log(messages)

        log_llm_call(
            call_type="Final Answer Synthesis (MuSiQue-V11-PathsHint + rag_qa_cot)",
            input_text=user_content,
            output_text=answer,
            context=context,
        )

        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        return answer

    async def answer_subquestion(
        self,
        sq: SubQuestion,
        decomposition: QueryDecomposition,
        is_final_sq: bool = False,
    ) -> Dict:
        try:
            actual_question = substitute_answers(sq.question, decomposition.subquestions)
            previous_context = self._build_simple_previous_context(sq, decomposition)

            passages, top_paths = await self.retrieve_for_query(actual_question)

            if self.verbose:
                print(f"\n   Retrieved {len(passages)} passages + {len(top_paths)} paths")

            answer = await self.generate_answer(
                actual_question,
                passages,
                top_paths,
                previous_context,
                decomposition.main_query,
                is_final_sq=is_final_sq,
            )

            sq.answer = answer
            sq.retrieved_passages = passages
            sq.retrieved_paths = top_paths

            return {
                'success': True,
                'answer': answer,
                'actual_question': actual_question,
                'passages': passages,
                'paths': top_paths,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def process_question(self, question: str) -> Dict:
        start_time = time.time()

        try:
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
                    'time': time.time() - start_time,
                }

            decomposition: QueryDecomposition = decomp_result['decomposition']

            if self.verbose:
                print(f"   Main query: {decomposition.main_query}")
                print(f"   Sub-questions: {len(decomposition.subquestions)}")

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

                    await self.answer_subquestion(sq, decomposition, is_final_sq=is_final)

            if self.verbose:
                print("\n[3] Generating final answer...")

            all_passages = self._collect_all_unique_passages(decomposition)
            final_answer = await self.generate_final_answer(
                decomposition.main_query,
                decomposition,
                all_passages,
            )

            elapsed = time.time() - start_time

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
                            'retrieved_passages': getattr(sq, 'retrieved_passages', []),
                            'retrieved_paths': getattr(sq, 'retrieved_paths', []),
                        }
                        for sq in decomposition.subquestions
                    ],
                },
                'num_passages': len(all_passages),
                'time': elapsed,
            }

        except Exception as e:
            elapsed = time.time() - start_time
            return {'success': False, 'error': str(e), 'time': elapsed}

    def close(self):
        if self.conn:
            self.conn.close()


async def _quick_smoke_test():
    import os
    from dotenv import load_dotenv

    load_dotenv()
    client = AsyncOpenAI(api_key=os.getenv('ALICE_OPENAI_KEY'), base_url=os.getenv('ALICE_CHAT_URL'))
    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6,
        bm25_index_path='MuSiQue/bm25_index',
        embeddings_path='MuSiQue/path_embeddings.npz',
    )
    pipeline = MuSiQueMultihopPipelineV11PathsHint(
        client=client,
        retriever=retriever,
        musique_path='MuSiQue/musique_sample_200.json',
        db_path='MuSiQue/metadata_v3.db',
        verbose=True,
    )

    with open('MuSiQue/musique_sample_200.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    q = data[0]['question']
    res = await pipeline.process_question(q)
    print(res['final_answer'] if res.get('success') else res.get('error'))
    pipeline.close()


if __name__ == '__main__':
    asyncio.run(_quick_smoke_test())
