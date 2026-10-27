import asyncio
import unittest
import tempfile
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

from new_multihop_pipeline_paths_hint_expansion import NewMultihopPipelineV12PathsHintExpansion
from run_simple_qa import answer_direct
from embedding_text_generator import EmbeddingTextGenerator, generate_embedding_texts_from_db
from hybrid_path_retriever import HybridPathRetriever
from setup_indices_v5 import validate_granularity
from query_decomposition import QueryDecomposition, SubQuestion, get_execution_order, substitute_answers


class ReleaseTests(unittest.TestCase):
    def test_db_fact_generation_keeps_source_ids_and_nested_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / 'metadata.db'
            with sqlite3.connect(db) as conn:
                conn.execute('CREATE TABLE metadata (doc_id TEXT, source_title TEXT, entity_title TEXT, metadata_json TEXT)')
                conn.execute('INSERT INTO metadata VALUES (?, ?, ?, ?)', ('0', 'Source', 'Entity', json.dumps({
                    'attributes': {'cast': [{'name': 'Alice'}, {'name': 'Bob'}]},
                    'relations': [{'relation': 'features', 'target': ['A', 'B']}],
                })))
            conn.close()
            facts = generate_embedding_texts_from_db(str(db), str(root / 'facts.json'))
            self.assertEqual(len(facts), 2)
            self.assertEqual({f['doc_id'] for f in facts}, {'0'})
            self.assertEqual({f['source_title'] for f in facts}, {'Source'})
            self.assertEqual({f['entity_title'] for f in facts}, {'Entity'})
            self.assertIn('Bob', facts[0]['text'])

    def test_weighted_rrf_uses_rank_not_raw_score(self):
        retriever = object.__new__(HybridPathRetriever)
        retriever.bm25_weight, retriever.dense_weight, retriever.rrf_k = 1.0, 1.3, 60
        retriever.search_bm25 = lambda query, k: [(0, 100000), (1, 1)]
        retriever.search_dense = AsyncMock(return_value=[(1, 0.9), (0, 0.8)])
        retriever.titles = ['A', 'B']
        retriever.key_paths = ['key', 'key']
        retriever.values = ['one', 'two']
        retriever.doc_ids = ['0', '1']
        retriever.source_titles = retriever.entity_titles = retriever.titles
        result = asyncio.run(retriever.search_hybrid('query', top_k=2))
        self.assertEqual(result[0]['doc_id'], '1')
        # Preserve the research implementation's zero-based rank convention.
        self.assertAlmostEqual(result[0]['score'], 1.0 / 61 + 1.3 / 60)

    def test_whole_list_preserved_and_legacy_mode_available(self):
        metadata = {'attributes': {'cast': [{'name': 'Alice', 'role': 'Lead'}, {'name': 'Bob', 'role': 'Guest'}]}}
        facts = EmbeddingTextGenerator().extract_embedding_texts('Film', metadata)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]['key_path'], 'cast')
        self.assertIn('Alice', facts[0]['text'])
        self.assertIn('Bob', facts[0]['text'])
        legacy = EmbeddingTextGenerator('legacy_items').extract_embedding_texts('Film', metadata)
        self.assertEqual(len(legacy), 2)

    def test_index_mode_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {k: root / k for k in ('embedding_texts', 'embeddings', 'bm25_index')}
            paths['embedding_texts'].write_text('[]')
            manifest = root / 'index_manifest.json'
            with self.assertRaises(ValueError):
                validate_granularity(paths, manifest, 'top_level')
            validate_granularity(paths, manifest, 'legacy_items')
            manifest.write_text(json.dumps({'fact_granularity': 'top_level'}))
            validate_granularity(paths, manifest, 'top_level')
            with self.assertRaises(ValueError):
                validate_granularity(paths, manifest, 'legacy_items')

    def test_main_and_table5_allocations_are_isolated(self):
        main = object.__new__(NewMultihopPipelineV12PathsHintExpansion)
        grid = object.__new__(NewMultihopPipelineV12PathsHintExpansion)
        main.configure_budget_profile('main')
        grid.configure_budget_profile('table5')
        self.assertEqual(main._PASSAGE_SEED_RERANK[5], (3, 2))
        self.assertEqual(main._PATH_SEED_RERANK[30], (20, 10))
        self.assertEqual(grid._PASSAGE_SEED_RERANK[5], (2, 3))
        self.assertEqual(grid._PATH_SEED_RERANK[30], (10, 20))
        self.assertEqual(grid._PATH_SEED_RERANK[0], (0, 0))

    def test_dependency_order_and_answer_substitution(self):
        first = SubQuestion('SQ1', 'Which county?', [], '')
        second = SubQuestion('SQ2', 'Which city in [SQ1_Answer]?', ['SQ1'], '')
        decomposition = QueryDecomposition('Main', 'multi', '', [second, first])
        self.assertEqual(get_execution_order(decomposition), [['SQ1'], ['SQ2']])
        first.answer = 'Pima'
        self.assertEqual(substitute_answers(second.question, [first]), 'Which city in Pima?')

    def test_paper_candidate_budget(self):
        pipeline = object.__new__(NewMultihopPipelineV12PathsHintExpansion)
        pipeline.retrieval_candidates = 300
        pipeline.fusion_method = 'rrf'
        pipeline.retriever = type('Retriever', (), {'search_hybrid': AsyncMock(return_value=[])})()
        asyncio.run(pipeline.retrieve_for_query_with_limits('question', 20, 50, 500))
        kwargs = pipeline.retriever.search_hybrid.call_args.kwargs
        self.assertEqual(kwargs['bm25_candidates'], 300)
        self.assertEqual(kwargs['dense_candidates'], 300)

    def test_simple_qa_bypasses_qd_and_reranking(self):
        facts = [{'doc_id': '1', 'key_path': 'year', 'value': '1875'}]
        passages = [{'doc_id': '1', 'title': 'Book'}]
        pipeline = type('Pipeline', (), {})()
        pipeline.retrieve_for_query_with_limits = AsyncMock(return_value=(passages, facts))
        async def generate(question, decomposition, retrieved):
            self.assertEqual(decomposition.subquestions, [])
            self.assertIs(decomposition._final_selected_paths, facts)
            self.assertIs(decomposition._final_selected_passages, passages)
            return '1875'
        pipeline.generate_final_answer = generate
        result = asyncio.run(answer_direct(pipeline, 'When?'))
        self.assertEqual(result['predicted_answer'], '1875')

    def test_retrieval_only_has_no_answer(self):
        pipeline = type('Pipeline', (), {})()
        pipeline.retrieve_for_query_with_limits = AsyncMock(return_value=([], []))
        result = asyncio.run(answer_direct(pipeline, 'When?', True))
        self.assertIsNone(result['predicted_answer'])

    def test_nested_attribute_and_grouped_relation_preserved(self):
        facts = EmbeddingTextGenerator().extract_embedding_texts('Book', {
            'attributes': {'author': {'name': 'Ion', 'nationality': 'Romanian'}},
            'relations': [{'relation': 'features', 'target': ['Goat', 'Wolf']}],
        })
        self.assertEqual(len(facts), 2)
        self.assertIn('Romanian', facts[0]['text'])
        self.assertIn('Goat', facts[1]['text'])
        self.assertIn('Wolf', facts[1]['text'])


if __name__ == '__main__':
    unittest.main()
