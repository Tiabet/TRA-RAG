#!/usr/bin/env python3
"""\
Multi-hop Pipeline (Paths-as-Hints + rag_qa_cot one-shot)
=======================================================

Based on `new_multihop_pipeline_paths_hint.py` (v11 paths-as-hints), but:
- Supports BOTH MuSiQue and HotpotQA dataset files.
- Applies `Prompt/rag_qa_cot.py` one-shot chat prompt template to BOTH:
    - sub-question answering
    - final main-query answering

Notes:
- Sub-question answering keeps dependency context (previous Q/A) and can include
    path hints, but is formatted as rag_qa_cot docs + Question + Thought:.
- Final answering prompt uses rag_qa_cot docs + Question + Thought: and:
    - does NOT include previous context
    - includes explicit instruction to use Metadata Paths (Hints)

Output remains compatible with:
- evaluate_retrieval.py (reads retrieved_passages titles)
- evaluate_mrqa.py (reads final_answer/predicted_answer)
- llm_evaluation.py (reads predicted_answer)

All LLM calls are logged via llm_logger.
"""

import asyncio
import copy
import json
import os
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

from Prompt.rag_qa_cot import fact_rag_qa_system, prompt_template_fact as RAG_QA_COT_PROMPT_TEMPLATE

from llm_logger import log_llm_call


COT_MAX_TOKENS = int(os.getenv("COT_MAX_TOKENS", "2048"))


def _extract_after_answer_marker(text: str) -> str:
    """Extract only the portion after the last 'Answer:' marker.

    This keeps CoT reasoning available in logs while storing a clean answer string
    for dependency substitution and evaluation.
    """
    if not text:
        return ""
    lower = text.lower()
    idx = lower.rfind("answer")
    if idx == -1:
        return text.strip()
    # Find the last occurrence of 'answer' followed by optional spaces and ':'
    # Do a simple scan from the end to be resilient to formatting.
    marker = "answer:"
    marker_idx = lower.rfind(marker)
    if marker_idx == -1:
        return text.strip()
    return text[marker_idx + len(marker):].strip()


class NewMultihopPipelineV11PathsHintCoT:
    """Pipeline using hybrid retrieval + original passages + top path hints (MuSiQue/HotpotQA) + rag_qa_cot one-shot."""

    def __init__(
        self,
        client: AsyncOpenAI,
        retriever: HybridPathRetriever,
        data_path: str = 'MuSiQue/musique_sample_200.json',
        db_path: str = 'MuSiQue/metadata_v4aligned.db',
        top_k_passages: int = 5,
        top_k_paths: int = 30,
        path_fetch_k: int = 50,
        verbose: bool = True,
        log_messages: bool = False,
        dataset_name: str = 'musique',
    ):
        self.client = client
        self.retriever = retriever
        self.db_path = db_path
        self.top_k_passages = top_k_passages
        self.top_k_paths = top_k_paths
        # For top-30 UNIQUE paths, we often need to fetch much more than 30 due to duplicates.
        self.path_fetch_k = max(path_fetch_k, top_k_paths * 10, top_k_passages * 10, 100)
        self.verbose = verbose
        self.log_messages = log_messages
        self.dataset_name = dataset_name

        self.original_passages, self.doc_id_passages = self._load_passage_indices(data_path)
        if self.verbose:
            print(f"[OK] Loaded {len(self.original_passages)} original passages")

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def _load_passage_indices(self, data_path: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        passages_by_title: Dict[str, str] = {}
        passages_by_doc_id: Dict[str, Dict[str, str]] = {}
        for item in data:
            sample_id = item.get('_id')
            for ctx_idx, (title, sentences) in enumerate(item.get('context', [])):
                full_text = ''.join(sentences).strip()
                if title not in passages_by_title:
                    passages_by_title[title] = full_text
                if sample_id:
                    doc_id = f"{sample_id}::ctx{ctx_idx}"
                    # Store title too, so downstream can display it without relying on path['source_title'].
                    passages_by_doc_id[str(doc_id)] = {"title": title, "text": full_text}

        return passages_by_title, passages_by_doc_id

    def get_original_passage(self, title: str) -> Optional[str]:
        return self.original_passages.get(title)

    def get_original_passage_by_doc_id(self, doc_id: Optional[str]) -> Optional[str]:
        if not doc_id:
            return None
        entry = self.doc_id_passages.get(str(doc_id))
        if entry:
            return entry.get('text')
        return None

    def get_title_by_doc_id(self, doc_id: Optional[str]) -> Optional[str]:
        if not doc_id:
            return None
        entry = self.doc_id_passages.get(str(doc_id))
        if entry:
            return entry.get('title')
        return None

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
        """Return (top_unique_passages, top_unique_paths_as_hints).

        Notes:
        - Passages are derived from the highest-scoring UNIQUE paths (doc_id) and deduplicated by doc_id.
        - Paths are deduplicated by (source_title, entity_title, key_path, value).
        """
        fetched_paths = await self.retriever.search_hybrid(
            query,
            top_k=self.path_fetch_k,
            bm25_candidates=max(50, self.path_fetch_k),
            dense_candidates=max(50, self.path_fetch_k),
        )

        # 1) Pick top-k UNIQUE paths in ranked order
        seen_path_keys = set()
        top_paths: List[Dict] = []
        for p in fetched_paths:
            if len(top_paths) >= self.top_k_paths:
                break
            source_title = p.get('source_title') or p.get('title') or ''
            entity_title = p.get('entity_title') or p.get('title') or ''
            key_path = p.get('key_path', '')
            value = p.get('value', '')
            path_key = (str(source_title), str(entity_title), str(key_path), str(value))
            if path_key in seen_path_keys:
                continue
            seen_path_keys.add(path_key)
            top_paths.append(p)

        # 2) Pick top-k unique passages derived from the top-scoring UNIQUE paths (doc_id-based).
        seen_doc_ids = set()
        passages: List[Dict] = []
        sorted_paths_for_passages = sorted(top_paths, key=self._safe_score, reverse=True)
        for path in sorted_paths_for_passages:
            if len(passages) >= self.top_k_passages:
                break

            doc_id = path.get('doc_id')
            if not doc_id:
                continue
            doc_id_str = str(doc_id)
            if doc_id_str in seen_doc_ids:
                continue

            original_passage = self.get_original_passage_by_doc_id(doc_id_str)
            if not original_passage:
                if self.verbose:
                    print(f"[WARN] No passage found for doc_id={doc_id_str} (during SQ passage selection)")
                continue

            seen_doc_ids.add(doc_id_str)

            entity_title = path.get('entity_title') or path.get('title')
            source_title = path.get('source_title') or entity_title
            title_from_doc = self.get_title_by_doc_id(doc_id_str)
            display_title = title_from_doc or str(source_title)
            metadata = self.get_full_metadata(str(entity_title), doc_id=doc_id_str)

            passages.append({
                'title': display_title,
                'source_title': str(source_title),
                'entity_title': str(entity_title),
                'doc_id': doc_id_str,
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
        # Keep behavior: return ranked paths, caller may choose to dedupe/trim.
        return fetched_paths

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

    @staticmethod
    def _build_all_subqa_context(decomposition: QueryDecomposition) -> str:
        context_parts = []
        for sq in decomposition.subquestions:
            if getattr(sq, 'answer', None):
                context_parts.append(f"{sq.id}: {sq.question}")
                context_parts.append(f"Answer: {sq.answer}")
                context_parts.append("")
        if context_parts:
            return "Previous Sub-Questions:\n" + "\n".join(context_parts)
        return ""

    def _collect_all_unique_passages(self, decomposition: QueryDecomposition) -> List[Dict]:
        seen_keys = set()
        unique_passages: List[Dict] = []

        for sq in decomposition.subquestions:
            if hasattr(sq, 'retrieved_passages') and sq.retrieved_passages:
                for passage in sq.retrieved_passages:
                    doc_id = passage.get('doc_id')
                    title = passage.get('title', '')
                    key = str(doc_id) if doc_id else str(title)
                    if key and key not in seen_keys:
                        seen_keys.add(key)
                        unique_passages.append(passage)

        return unique_passages

    @staticmethod
    def _path_dedupe_key(p: Dict) -> Tuple[str, str, str, str]:
        source_title = p.get('source_title') or p.get('title') or ''
        entity_title = p.get('entity_title') or p.get('title') or ''
        key_path = p.get('key_path', '')
        value = p.get('value', '')
        return (str(source_title), str(entity_title), str(key_path), str(value))

    def _collect_all_unique_paths(self, decomposition: QueryDecomposition) -> List[Dict]:
        seen = set()
        unique_paths: List[Dict] = []

        for sq in decomposition.subquestions:
            paths = getattr(sq, 'retrieved_paths', None)
            if not paths:
                continue
            for p in paths:
                key = self._path_dedupe_key(p)
                if key in seen:
                    continue
                seen.add(key)
                unique_paths.append(p)

        return unique_paths

    @staticmethod
    def _safe_score(p: Dict) -> float:
        try:
            s = p.get('score', None)
            return float(s) if s is not None else float('-inf')
        except Exception:
            return float('-inf')

    def _select_top_paths_and_passages_from_decomposition(
        self,
        decomposition: QueryDecomposition,
        top_paths_k: int = 30,
        top_passages_k: int = 5,
    ) -> Tuple[List[Dict], List[Dict]]:
        all_paths = self._collect_all_unique_paths(decomposition)
        sorted_paths = sorted(all_paths, key=self._safe_score, reverse=True)
        top_paths = sorted_paths[:top_paths_k]

        top_path_passages: List[Dict] = []
        seen_doc_ids = set()
        for p in top_paths:
            if len(top_path_passages) >= top_passages_k:
                break
            doc_id = p.get('doc_id')
            if not doc_id:
                continue
            doc_id_str = str(doc_id)
            if doc_id_str in seen_doc_ids:
                continue
            passage_text = self.get_original_passage_by_doc_id(doc_id_str)
            if not passage_text:
                if self.verbose:
                    print(f"[WARN] No passage found for doc_id={doc_id_str} (from high-score path)")
                continue
            seen_doc_ids.add(doc_id_str)
            title_from_doc = self.get_title_by_doc_id(doc_id_str) or (p.get('source_title') or p.get('title') or '')
            top_path_passages.append({
                'title': str(title_from_doc),
                'doc_id': doc_id_str,
                'original_passage': passage_text,
                'metadata': None,
            })

        return top_paths, top_path_passages

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

    @staticmethod
    def _build_prompt_user_rag_qa_template(
        passages: List[Tuple[str, str]],
        question: str,
        *,
        facts_text: str = "",
        previous_context_text: str = "",
    ) -> str:
        """Build user content matching Prompt/rag_qa_cot.py one-shot format.

        Important formatting rules:
        - Only REAL passages are formatted with `Wikipedia Title:`.
        - Facts/hints and previous context are shown as explicit sections (no Wikipedia Title).
        """

        parts: List[str] = []
        # Make the instruction visible in FULL PROMPT.
        parts.append("---Instruction---\n" + fact_rag_qa_system.strip())
        if previous_context_text and previous_context_text.strip():
            parts.append("---Previous Context---\n" + previous_context_text.strip())

        # Passages first (user request), then facts.
        docs = ""
        for title, text in passages:
            docs += f"Wikipedia Title: {title}\n{text}\n"
        parts.append(docs.strip())

        if facts_text and facts_text.strip():
            parts.append("---Facts---\n" + facts_text.strip())

        return "\n\n".join([p for p in parts if p.strip()]) + f"\n\nQuestion: {question}\nThought: "

    async def generate_answer(
        self,
        question: str,
        passages: List[Dict],
        top_paths: List[Dict],
        previous_context: str,
        main_query: str,
        is_final_sq: bool = False,
    ) -> str:
        # SQ answering: NO previous context (user request). Only passages + facts/hints.
        docs: List[Tuple[str, str]] = []
        for p in passages:
            title = p.get('title') or ''
            original = p.get('original_passage') or ''
            docs.append((str(title), str(original)))

        facts_text = ""
        if top_paths:
            facts_text = self._format_paths_as_hints(top_paths)

        user_content = self._build_prompt_user_rag_qa_template(
            docs,
            question,
            facts_text=facts_text,
            previous_context_text="",
        )

        messages = self._build_rag_qa_cot_messages(user_content)
        response = await self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=COT_MAX_TOKENS,
        )

        raw = response.choices[0].message.content.strip()
        answer = _extract_after_answer_marker(raw)

        context = {
            "question": question,
            "main_query": main_query,
            "is_final_sq": is_final_sq,
            "num_passages": len(passages),
            "num_paths": len(top_paths),
        }
        # Always include enough template info to verify one-shot/system/assistant messages.
        try:
            context["system_message"] = messages[0].get("content", "")
            context["one_shot_user"] = messages[1].get("content", "")
            context["one_shot_assistant"] = messages[2].get("content", "")
        except Exception:
            pass
        if self.log_messages:
            context["chat_messages"] = self._messages_for_log(messages)

        log_llm_call(
            call_type=f"Subquestion Answering ({self.dataset_name}-V11-PathsHint + rag_qa_cot)",
            input_text=user_content,
            output_text=raw,
            context=context,
        )
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

    async def generate_final_answer(
        self,
        main_query: str,
        decomposition: QueryDecomposition,
        all_passages: List[Dict],
    ) -> str:
        # IMPORTANT: Do NOT re-retrieve for final answering.
        # Final answer uses top passages derived from SQ evidence only.
        top_paths, top_path_passages = self._select_top_paths_and_passages_from_decomposition(
            decomposition,
            top_paths_k=30,
            top_passages_k=5,
        )

        docs: List[Tuple[str, str]] = []
        for p in top_path_passages:
            title = p.get('title') or ''
            original = p.get('original_passage') or ''
            docs.append((str(title), str(original)))

        previous_context_text = self._build_all_subqa_context(decomposition)
        facts_text = self._format_paths_as_hints(top_paths) if top_paths else ""

        user_content = self._build_prompt_user_rag_qa_template(
            docs,
            main_query,
            facts_text=facts_text,
            previous_context_text=previous_context_text,
        )

        messages = self._build_rag_qa_cot_messages(user_content)

        response = await self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=COT_MAX_TOKENS,
        )

        raw = response.choices[0].message.content.strip()
        answer = _extract_after_answer_marker(raw)

        context = {
            "main_query": main_query,
            "num_passages": len(top_path_passages),
            "num_paths": len(top_paths),
            "num_top_path_passages": len(top_path_passages),
        }
        try:
            context["system_message"] = messages[0].get("content", "")
            context["one_shot_user"] = messages[1].get("content", "")
            context["one_shot_assistant"] = messages[2].get("content", "")
        except Exception:
            pass
        if self.log_messages:
            context["chat_messages"] = self._messages_for_log(messages)

        log_llm_call(
            call_type=f"Final Answer Synthesis ({self.dataset_name}-V11-PathsHint + rag_qa_cot)",
            input_text=user_content,
            output_text=raw,
            context=context,
        )
        return answer

    async def answer_subquestion(
        self,
        sq: SubQuestion,
        decomposition: QueryDecomposition,
        is_final_sq: bool = False,
    ) -> Dict:
        try:
            actual_question = substitute_answers(sq.question, decomposition.subquestions)
            # SQ answering: previous context is intentionally NOT used (user request).
            previous_context = ""

            passages, top_paths = await self.retrieve_for_query(actual_question)

            if self.verbose:
                print(f"\n   Retrieved {len(passages)} passages + {len(top_paths)} facts")

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
                # Backward compatibility
                'paths': top_paths,
                # Preferred naming
                'facts': top_paths,
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

            # For reporting, we consider passages used in FINAL answering only.
            final_paths, final_passages = self._select_top_paths_and_passages_from_decomposition(
                decomposition,
                top_paths_k=30,
                top_passages_k=5,
            )
            final_answer = await self.generate_final_answer(
                decomposition.main_query,
                decomposition,
                [],
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
                            # Backward compatibility
                            'retrieved_paths': getattr(sq, 'retrieved_paths', []),
                            # Preferred naming
                            'retrieved_facts': getattr(sq, 'retrieved_paths', []),
                        }
                        for sq in decomposition.subquestions
                    ],
                },
                'num_passages': len(final_passages),
                # Backward compatibility
                'num_paths': len(final_paths),
                # Preferred naming
                'num_facts': len(final_paths),
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
    from llm_logger import init_logger, finalize_log

    load_dotenv()
    init_logger()
    client = AsyncOpenAI(api_key=os.getenv('ALICE_OPENAI_KEY'), base_url=os.getenv('ALICE_CHAT_URL'))
    dataset = os.getenv('DATASET', 'musique').lower()
    if dataset == 'hotpotqa':
        data_path = 'HotpotQA/hotpotqa_sample_200.json'
        db_path = 'HotpotQA/metadata_v4aligned.db'
        bm25_index_path = 'HotpotQA/bm25_index_v4aligned'
        embeddings_path = 'HotpotQA/path_embeddings_v4aligned.npz'
    else:
        dataset = 'musique'
        data_path = 'MuSiQue/musique_sample_200.json'
        db_path = 'MuSiQue/metadata_v4aligned.db'
        bm25_index_path = 'MuSiQue/bm25_index_v4aligned'
        embeddings_path = 'MuSiQue/path_embeddings_v4aligned.npz'

    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6,
        bm25_index_path=bm25_index_path,
        embeddings_path=embeddings_path,
    )
    pipeline = NewMultihopPipelineV11PathsHintCoT(
        client=client,
        retriever=retriever,
        data_path=data_path,
        db_path=db_path,
        verbose=True,
        dataset_name=dataset,
    )

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    q = data[0]['question']
    res = await pipeline.process_question(q)
    print(res['final_answer'] if res.get('success') else res.get('error'))
    pipeline.close()
    finalize_log()


if __name__ == '__main__':
    asyncio.run(_quick_smoke_test())


# Backward-compatible alias (older scripts import this name)
MuSiQueMultihopPipelineV11PathsHint = NewMultihopPipelineV11PathsHintCoT
