#!/usr/bin/env python3
"""\
New Multi-hop Pipeline v11 (Paths-as-Hints)
==========================================
Extends v3 (original-passages answering) with an additional signal:

- Retrieval: RRF hybrid (BM25+dense) over metadata paths.
- For each sub-question:
  - Select top-N paths (default: 10) as *strong hints*
  - Select top-K unique original passages (default: 3) using source_title
  - Answer using BOTH: passages(3) + paths(10)

Output remains compatible with:
- evaluate_retrieval.py (reads retrieved_passages titles)
- evaluate_mrqa.py (reads final_answer/predicted_answer)
- llm_evaluation.py (reads predicted_answer)

All LLM calls are logged via llm_logger.
"""

import asyncio
import json
import time
import sqlite3
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from query_decomposition import (
    decompose_query,
    QueryDecomposition,
    SubQuestion,
    substitute_answers,
    get_execution_order,
)
from hybrid_path_retriever import HybridPathRetriever

from Prompt.answer import (
    DETAILED_SUBQUESTION_ANSWERING_PROMPT,
    FINAL_SUBQUESTION_ANSWERING_PROMPT,
)
from Prompt.subquestion_answering_prompt import FINAL_ANSWER_SYNTHESIS_PROMPT

from llm_logger import log_llm_call


class NewMultihopPipelineV11PathsHint:
    """Pipeline using hybrid retrieval + original passages + top path hints for SQ answering."""

    def __init__(
        self,
        client: AsyncOpenAI,
        retriever: HybridPathRetriever,
        hotpotqa_path: str = 'HotpotQA/hotpotqa_sample_200.json',
        db_path: str = 'HotpotQA/metadata_v4aligned.db',
        top_k_passages: int = 5,
        top_k_paths: int = 30,
        path_fetch_k: int = 50,
        verbose: bool = True,
    ):
        self.client = client
        self.retriever = retriever
        self.db_path = db_path
        self.top_k_passages = top_k_passages
        self.top_k_paths = top_k_paths
        # For top-k UNIQUE paths, we often need to fetch much more than k due to duplicates.
        # Store the effective fetch-k so metadata/logging matches actual behavior.
        self.path_fetch_k_input = path_fetch_k
        self.path_fetch_k = max(path_fetch_k, top_k_paths * 10, top_k_passages * 10, 100)
        self.verbose = verbose

        self.original_passages, self.doc_id_passages = self._load_passage_indices(hotpotqa_path)
        if self.verbose:
            print(f"[OK] Loaded {len(self.original_passages)} original passages")

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def _load_passage_indices(self, hotpotqa_path: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
        with open(hotpotqa_path, 'r', encoding='utf-8') as f:
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
        - Passages are derived from the highest-scoring paths (doc_id) and deduplicated by doc_id.
        - Paths are deduplicated by (source_title, entity_title, key_path, value).
        """

        return await self.retrieve_for_query_with_limits(
            query=query,
            top_k_passages=self.top_k_passages,
            top_k_paths=self.top_k_paths,
            path_fetch_k=self.path_fetch_k,
        )

    async def retrieve_for_query_with_limits(
        self,
        query: str,
        top_k_passages: int,
        top_k_paths: int,
        path_fetch_k: int,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Retrieve using explicit limits (used for final-answer override like top-30 paths)."""
        # For top-30 UNIQUE paths, we often need to fetch much more than 30 due to duplicates.
        fetch_k = max(path_fetch_k, top_k_paths * 10, top_k_passages * 10, 100)

        fetched_paths = await self.retriever.search_hybrid(
            query,
            top_k=fetch_k,
            bm25_candidates=max(50, fetch_k),
            dense_candidates=max(50, fetch_k),
        )

        # 1) Pick top-k UNIQUE paths in ranked order
        seen_path_keys = set()
        top_paths: List[Dict] = []
        for p in fetched_paths:
            if len(top_paths) >= top_k_paths:
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

        # Passages are selected from the top-scoring UNIQUE paths.
        # This ties passage selection strictly to path scores.
        sorted_paths_for_passages = sorted(top_paths, key=self._safe_score, reverse=True)
        for path in sorted_paths_for_passages:
            if len(passages) >= top_k_passages:
                break

            doc_id = path.get('doc_id')
            if not doc_id:
                continue
            doc_id_str = str(doc_id)
            if doc_id_str in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id_str)

            entity_title = path.get('entity_title') or path.get('title')
            source_title = path.get('source_title') or entity_title

            # IMPORTANT: When pulling passages from paths, use doc_id mapping (not title matching).
            original_passage = self.get_original_passage_by_doc_id(doc_id_str)
            if not original_passage:
                if self.verbose:
                    print(f"[WARN] No passage found for doc_id={doc_id_str} (during SQ passage selection)")
                continue
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
                # Fallback to metadata if original not available
                if p.get('metadata'):
                    metadata = p['metadata']
                    parts = [f"[{i}] {title}"]
                    excluded_keys = {'title'}
                    for key, value in metadata.items():
                        if key in excluded_keys or not value:
                            continue
                        value_str = str(value) if not isinstance(value, (dict, list)) else json.dumps(value, ensure_ascii=False)
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
            "---Top Passages from High-Score Paths (TOP-5 by doc_id)---\n"
            f"{passages_text}"
        )

        prompt_template = FINAL_SUBQUESTION_ANSWERING_PROMPT if is_final_sq else DETAILED_SUBQUESTION_ANSWERING_PROMPT

        prompt = prompt_template.replace("{{subquestion}}", question)
        prompt = prompt.replace("{{passages}}", combined_info)
        prompt = prompt.replace("{{previous_context}}", previous_context if previous_context else "None")
        prompt = prompt.replace("{{main_query}}", main_query)

        response = await self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise question answering system."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )

        answer = response.choices[0].message.content.strip()

        log_llm_call(
            call_type="Subquestion Answering (V11-PathsHint)",
            input_text=prompt,
            output_text=answer,
            context={
                "question": question,
                "main_query": main_query,
                "is_final_sq": is_final_sq,
                "num_passages": len(passages),
                "num_paths": len(top_paths),
            },
        )

        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        return answer

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

        # IMPORTANT: Do NOT re-retrieve for final answering.
        # Final answer uses:
        # - top-30 paths by score (unique; reused from SQs)
        # - top-5 passages derived from those paths by doc_id (NOT all SQ passages)
        final_paths, top_path_passages = self._select_top_paths_and_passages_from_decomposition(
            decomposition,
            top_paths_k=30,
            top_passages_k=5,
        )
        paths_text = self._format_paths_as_hints(final_paths)
        top_path_passages_text = self._format_passages_original(top_path_passages)
        combined_info = (
            "---Top Retrieved Metadata Paths (TOP-30 by score, UNIQUE; reused from SQs)---\n"
            "The paths below are strong hints for where the answer might be found. "
            "Use them to focus your reading of the passages, but do NOT treat them as guaranteed truth.\n\n"
            f"{paths_text}\n\n"
            "---Top Passages from High-Score Paths (TOP-5 by doc_id)---\n"
            f"{top_path_passages_text}"
        )

        prompt = FINAL_ANSWER_SYNTHESIS_PROMPT.replace("{{main_question}}", main_query)
        prompt = prompt.replace("{{subquestion_chain}}", subquestion_chain)
        # Inject combined paths+passages into the {{passages}} slot.
        prompt = prompt.replace("{{passages}}", combined_info)

        response = await self.client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise question answering system. Give short, direct answers."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=100,
        )

        answer = response.choices[0].message.content.strip()

        log_llm_call(
            call_type="Final Answer Synthesis (V11-PathsHint)",
            input_text=prompt,
            output_text=answer,
            context={
                "main_query": main_query,
                "num_passages": len(top_path_passages),
                "num_paths": len(final_paths),
                "num_top_path_passages": len(top_path_passages),
            },
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
            setattr(sq, 'actual_question', actual_question)
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
                # Final-only retrieval artifacts (doc_id-based), for @k evaluation.
                'final_retrieved_passages': [
                    {
                        'doc_id': p.get('doc_id'),
                        'title': p.get('title'),
                    }
                    for p in (final_passages or [])
                ],
                'final_retrieved_paths': [
                    {
                        'doc_id': p.get('doc_id'),
                    }
                    for p in (final_paths or [])
                ],
                'decomposition': {
                    'main_query': decomposition.main_query,
                    'subquestions': [
                        {
                            'id': sq.id,
                            'question': sq.question,
                            'actual_question': getattr(sq, 'actual_question', None),
                            'answer': sq.answer,
                            'depends_on': sq.depends_on,
                            'retrieved_passages': getattr(sq, 'retrieved_passages', []),
                            'retrieved_paths': getattr(sq, 'retrieved_paths', []),
                        }
                        for sq in decomposition.subquestions
                    ],
                },
                'num_passages': len(final_passages),
                'num_paths': len(final_paths),
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
    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6,
        bm25_index_path='MuSiQue/bm25_index_v4aligned',
        embeddings_path='MuSiQue/path_embeddings_v4aligned.npz',
    )
    pipeline = NewMultihopPipelineV11PathsHint(
        client=client,
        retriever=retriever,
        hotpotqa_path='MuSiQue/musique_sample_200.json',
        db_path='MuSiQue/metadata_v4aligned.db',
        verbose=True,
    )

    with open('MuSiQue/musique_sample_200.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    q = data[0]['question']
    res = await pipeline.process_question(q)
    print(res['final_answer'] if res.get('success') else res.get('error'))
    pipeline.close()
    finalize_log()


if __name__ == '__main__':
    asyncio.run(_quick_smoke_test())
