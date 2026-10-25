import asyncio
import unittest
from unittest.mock import AsyncMock

from new_multihop_pipeline_paths_hint_expansion import NewMultihopPipelineV12PathsHintExpansion
from run_simple_qa import answer_direct
from embedding_text_generator import EmbeddingTextGenerator


class ReleaseTests(unittest.TestCase):
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
