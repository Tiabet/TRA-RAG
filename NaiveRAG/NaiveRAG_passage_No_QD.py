#!/usr/bin/env python3
"""
Naive RAG - Passage Level (No Query Decomposition)
==================================================
Retrieves passages directly using vector embeddings and generates answers.

Usage:
    python NaiveRAG/NaiveRAG_passage_No_QD.py --dataset hotpotqa --k 5
    python NaiveRAG/NaiveRAG_passage_No_QD.py --dataset musique --k 5
"""

import asyncio
import json
import os
import sys
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NaiveRAG.naive_passage_retriever import NaivePassageRetriever
from llm_logger import init_logger, finalize_log, log_llm_call

from Prompt.answer_prompt import NAIVE_RAG_FINAL_ANSWER_PROMPT

async def process_question(client, retriever, item, k):
    question = item['question']
    
    # Retrieve
    results = await retriever.search(question, k=k)
    
    # Format passages
    passage_text = ""
    for i, res in enumerate(results, 1):
        passage_text += f"[{i}] {res['title']}\n{res['text']}\n\n"
        
    prompt = NAIVE_RAG_FINAL_ANSWER_PROMPT.replace("{{passages}}", passage_text.strip())
    prompt = prompt.replace("{{question}}", question)
    
    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini", # Or whatever model is standard
        messages=[
            {"role": "system", "content": "You are a precise question answering system."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=50
    )
    
    answer = response.choices[0].message.content.strip()

    log_llm_call(
        call_type="Answer Generation (NaiveRAG No_QD)",
        input_text=prompt,
        output_text=answer,
        context={
            "question": question,
            "k": k,
            "num_passages": len(results),
            "passages": [r['title'] for r in results],
        },
    )
    
    qid = item.get('_id') or item.get('id')

    final_retrieved_passages = [
        {
            'doc_id': r.get('doc_id'),
            'title': r.get('title'),
            'score': r.get('score'),
        }
        for r in results
    ]

    return {
        'id': qid,
        'question': question,
        'gold_answer': item['answer'],
        'predicted_answer': answer,
        'answer_aliases': item.get('answer_aliases', []),
        'final_retrieved_passages': final_retrieved_passages,
        'retrieved_passages': [{'doc_id': r.get('doc_id'), 'title': r.get('title')} for r in results],
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5, help='Number of passages to retrieve (fixed at 5 by default)')
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['hotpotqa', 'musique'])
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
    print(f"Processing {len(data)} questions with K={args.k}...")
    
    start_time = time.time()
    
    # Process in batches for concurrency
    batch_size = 50  # Adjust based on rate limits
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        tasks = [process_question(chat_client, retriever, item, args.k) for item in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        print(f"Processed {min(i+batch_size, len(data))}/{len(data)}")
            
    total_time = time.time() - start_time
    
    # Save Results
    output_file = f'Results/NaiveRAG/NaiveRAG_passage_No_QD_{args.dataset}_k{args.k}.json'
    os.makedirs('Results/NaiveRAG', exist_ok=True)
    
    output = {
        'config': {
            'pipeline': 'NaiveRAG_No_QD',
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
