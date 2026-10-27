"""Direct TRA retrieval and answering: no QD and no final reranking."""
import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from hybrid_path_retriever import HybridPathRetriever
from llm_provider import create_async_chat_client, detect_provider
from new_multihop_pipeline_paths_hint_expansion import NewMultihopPipelineV12PathsHintExpansion
from query_decomposition import QueryDecomposition


async def answer_direct(pipeline, question, retrieval_only=False):
    passages, facts = await pipeline.retrieve_for_query_with_limits(
        question, top_k_passages=5, top_k_paths=30, path_fetch_k=300
    )
    answer = None
    if not retrieval_only:
        decomposition = QueryDecomposition(
            main_query=question, question_type='simple', reasoning='', subquestions=[]
        )
        # Reuse the published answering prompt, supplying the direct RRF context.
        # Supplying both lists bypasses the multi-hop final selection routine.
        decomposition._final_selected_paths = facts
        decomposition._final_selected_passages = passages
        answer = await pipeline.generate_final_answer(question, decomposition, passages)
    return {
        'success': True, 'final_answer': answer, 'predicted_answer': answer,
        'retrieval_only': retrieval_only,
        'final_retrieved_passages': [
            {'doc_id': p['doc_id'], 'title': p.get('title')} for p in passages
        ],
        'final_retrieved_paths': facts,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['nq_rear', 'popqa', 'hotpot', 'musique', '2wiki', 'lveval'], required=True)
    p.add_argument('--qa_path')
    p.add_argument('--corpus_path')
    p.add_argument('--artifact_dir')
    p.add_argument('--output')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--retrieval_only', action='store_true', help='Skip chat; embeddings may call an API')
    return p.parse_args()


async def run(args):
    load_dotenv()
    inputs = {
        'nq_rear': ('NQ_REAR', 'nq_rear_qa.json', 'nq_rear_corpus_for_pipeline.json'),
        'popqa': ('POPQA', 'popqa_qa.json', 'popqa_corpus_for_pipeline.json'),
        'hotpot': ('HotpotQA', 'hotpotqa.json', 'hotpotqa.json'),
        'musique': ('MuSiQue', 'musique.json', 'musique.json'),
        '2wiki': ('2WikiMultihopQA', '2wiki_qa.json', '2wikimultihopqa.json'),
        'lveval': ('LVEVAL', 'lveval_qa_compact.json', 'lveval_corpus_for_pipeline.json'),
    }
    folder, qa_name, corpus_name = inputs[args.dataset]
    directory = Path(args.artifact_dir or folder)
    qa = Path(args.qa_path or directory / qa_name)
    corpus = Path(args.corpus_path or directory / corpus_name)
    output = Path(args.output or f'Results/{args.dataset}_simple.json')
    data = json.loads(qa.read_text(encoding='utf-8'))
    if args.limit:
        data = data[:args.limit]
    cfg = detect_provider()
    client = None if args.retrieval_only else create_async_chat_client(cfg)
    retriever = HybridPathRetriever(
        bm25_index_path=str(directory / 'bm25_index_v5'),
        embeddings_path=str(directory / 'path_embeddings_v5.npz'),
        bm25_weight=1.0, dense_weight=1.3,
    )
    pipeline = NewMultihopPipelineV12PathsHintExpansion(
        client=client, retriever=retriever, hotpotqa_path=str(corpus),
        db_path=str(directory / 'metadata_v5.db'), use_previous_context=False,
        chat_model=cfg.chat_model, retrieval_candidates=300, verbose=False,
    )
    results = []
    try:
        for i, item in enumerate(data):
            result = await answer_direct(pipeline, item['question'], args.retrieval_only)
            results.append({
                'id': item.get('id') or item.get('_id') or str(i),
                'question': item['question'], 'gold_answer': item.get('answer'),
                'answer_aliases': item.get('answer_aliases', []), **result,
            })
            print(f'{i + 1}/{len(data)}')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    finally:
        pipeline.close()
        if client is not None:
            await client.close()


if __name__ == '__main__':
    asyncio.run(run(parse_args()))
