#!/usr/bin/env python3
"""
Naive RAG - Passage Level (With Query Decomposition)
====================================================
Uses Query Decomposition to retrieve passages for sub-questions,
then synthesizes the final answer using unique retrieved passages.

Usage:
    python NaiveRAG/NaiveRAG_passage_QD.py --dataset hotpotqa --k 5
    python NaiveRAG/NaiveRAG_passage_QD.py --dataset musique --k 5
"""

import asyncio
import json
import os
import sys
import argparse
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NaiveRAG.naive_passage_retriever import NaivePassageRetriever
from query_decomposition import decompose_query, substitute_answers
from Prompt.answer_prompt import (
    DETAILED_SUBQUESTION_ANSWERING_PROMPT,
    FINAL_ANSWER_SYNTHESIS_PROMPT,
)
from llm_logger import log_llm_call, init_logger, finalize_log

async def answer_subquestion(client, retriever, sq, decomposition, previous_context, k: int):
    # Substitute placeholders
    actual_question = substitute_answers(sq.question, decomposition.subquestions)
    
    # Retrieve Top-k (default: 5)
    results = await retriever.search(actual_question, k=k)
    
    # Format passages
    passage_text = ""
    for i, res in enumerate(results, 1):
        passage_text += f"[{i}] {res['title']}\n{res['text']}\n\n"
        
    full_prompt = DETAILED_SUBQUESTION_ANSWERING_PROMPT
    full_prompt = full_prompt.replace("{{main_query}}", decomposition.main_query)
    full_prompt = full_prompt.replace("{{previous_context}}", previous_context if previous_context else "None")
    full_prompt = full_prompt.replace("{{passages}}", passage_text.strip() if passage_text else "No passages retrieved.")
    full_prompt = full_prompt.replace("{{subquestion}}", actual_question)
    
    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise question answering system."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.0,
        max_tokens=50
    )
    
    answer = response.choices[0].message.content.strip()
    
    # Log LLM call for debugging
    log_llm_call(
        call_type="SubQuestion Answering (NaiveRAG)",
        input_text=full_prompt,
        output_text=answer,
        context={
            "subquestion": actual_question,
            "num_passages": len(results),
            "passages": [r['title'] for r in results]
        }
    )
    
    return answer, results

async def process_question(client, retriever, item, k: int):
    question = item['question']
    
    # 1. Decompose
    decomposition_result = await decompose_query(client, question)
    if not decomposition_result or not decomposition_result.get('success'):
        return {'question': question, 'error': 'Decomposition failed'}
    
    decomposition = decomposition_result['decomposition']
        
    all_passages = {} # doc_id -> {title, text, best_score}
    
    # 2. Process Sub-questions
    for sq in decomposition.subquestions:
        # Build context
        prev_context = ""
        if sq.depends_on:
            for dep_id in sq.depends_on:
                dep_sq = decomposition.get_subquestion(dep_id)
                if dep_sq and dep_sq.answer:
                    prev_context += f"Q: {dep_sq.question}\nA: {dep_sq.answer}\n\n"
        
        answer, results = await answer_subquestion(client, retriever, sq, decomposition, prev_context, k=k)
        sq.answer = answer
        sq.retrieved_passages = results
        
        # Collect passages (keep best score per doc_id)
        for res in results:
            doc_id = str(res.get('doc_id')) if res.get('doc_id') is not None else ''
            if not doc_id:
                continue
            title = res.get('title')
            score = float(res.get('score', 0.0))
            if doc_id not in all_passages or score > all_passages[doc_id]['best_score']:
                all_passages[doc_id] = {'title': title, 'text': res['text'], 'best_score': score}
            
    # 3. Final Answer
    # Final Answer: use top-5 passages among those collected from sub-questions
    ranked = sorted(all_passages.items(), key=lambda kv: kv[1]['best_score'], reverse=True)
    top_passages = ranked[:5]

    unique_passages_text = ""
    final_retrieved_passages = []
    for i, (doc_id, payload) in enumerate(top_passages, 1):
        title = payload.get('title') or ''
        unique_passages_text += f"[{i}] {title}\n{payload['text']}\n\n"
        final_retrieved_passages.append({
            'doc_id': str(doc_id),
            'title': title,
            'score': float(payload.get('best_score', 0.0)),
        })
        
    # Format chain
    chain_text = ""
    for sq in decomposition.subquestions:
        chain_text += f"{sq.id}: {sq.question}\nAnswer: {sq.answer}\n\n"
        
    prompt = FINAL_ANSWER_SYNTHESIS_PROMPT.replace("{{main_question}}", question)
    prompt = prompt.replace("{{subquestion_chain}}", chain_text)
    prompt = prompt.replace("{{passages}}", unique_passages_text)
    
    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise question answering system."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=50
    )
    
    final_answer = response.choices[0].message.content.strip()
    
    # Log Final Answer
    log_llm_call(
        call_type="Final Answer Synthesis (NaiveRAG)",
        input_text=prompt,
        output_text=final_answer,
        context={
            "main_query": question,
            "num_unique_passages": len(all_passages)
        }
    )
    
    qid = item.get('_id') or item.get('id')

    return {
        'id': qid,
        'question': question,
        'gold_answer': item['answer'],
        'predicted_answer': final_answer,
        'answer_aliases': item.get('answer_aliases', []),
        'final_retrieved_passages': final_retrieved_passages,
        'decomposition': [sq.to_dict() for sq in decomposition.subquestions]
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['hotpotqa', 'musique'])
    parser.add_argument('--k', type=int, default=5, help='Number of passages to retrieve per sub-question (default: 5)')
    args = parser.parse_args()
    
    load_dotenv()
    init_logger()
    
    # Config
    if args.dataset == 'hotpotqa':
        data_path = 'HotpotQA/hotpotqa_sample_200_corpus_idx.json'
        cache_path = 'HotpotQA/passage_embeddings_sample_200.npz'
    else:
        data_path = 'MuSiQue/musique_sample_200_corpus_idx.json'
        cache_path = 'MuSiQue/passage_embeddings_sample_200.npz'
        
    # Chat Client (for answer generation)
    chat_client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    # Embed Client (for embedding generation)
    embed_client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_EMBED_URL')
    )
    
    retriever = NaivePassageRetriever(embed_client, data_path, cache_path)
    await retriever.initialize()
    
    # Load Data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    results = []
    print(f"Processing {len(data)} questions with QD (k={args.k})...")
    
    import time
    start_time = time.time()
    
    # Process in batches for concurrency
    batch_size = 50  # Adjust based on rate limits
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        tasks = [process_question(chat_client, retriever, item, k=args.k) for item in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        print(f"Processed {min(i+batch_size, len(data))}/{len(data)}")
            
    total_time = time.time() - start_time
    
    # Save Results
    output_file = f'Results/NaiveRAG/NaiveRAG_passage_QD_{args.dataset}_k{args.k}.json'
    os.makedirs('Results/NaiveRAG', exist_ok=True)
    
    output = {
        'config': {
            'pipeline': 'NaiveRAG_QD',
            'dataset': args.dataset,
            'k': args.k,
            'model': 'gpt-4o-mini'
        },
        'summary': {
            'total_questions': len(data),
            'total_time': total_time,
            'avg_time_per_question': total_time / len(data) if data else 0
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        
    print(f"Saved results to {output_file}")
    log_path = finalize_log()
    print(f"[LOG] {log_path}")

if __name__ == "__main__":
    asyncio.run(main())
