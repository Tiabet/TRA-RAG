#!/usr/bin/env python3
"""
Naive RAG - Passage Level (With Query Decomposition)
====================================================
Uses Query Decomposition to retrieve passages for sub-questions,
then synthesizes the final answer using unique retrieved passages.

Usage:
    python NaiveRAG/NaiveRAG_passage_QD.py --dataset hotpotqa
    python NaiveRAG/NaiveRAG_passage_QD.py --dataset musique
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
from Prompt.answer import DETAILED_SUBQUESTION_ANSWERING_PROMPT
from Prompt.subquestion_answering_prompt import FINAL_ANSWER_SYNTHESIS_PROMPT
from llm_logger import log_llm_call, init_logger

async def answer_subquestion(client, retriever, sq, decomposition, previous_context):
    # Substitute placeholders
    actual_question = substitute_answers(sq.question, decomposition.subquestions)
    
    # Retrieve Top-3
    results = await retriever.search(actual_question, k=3)
    
    # Format passages
    passage_text = ""
    for i, res in enumerate(results, 1):
        passage_text += f"[{i}] {res['title']}\n{res['text']}\n\n"
        
    # Prompt
    prompt = DETAILED_SUBQUESTION_ANSWERING_PROMPT.replace("{{main_query}}", decomposition.main_query)
    
    # We need to inject the specific subquestion and passages into the prompt
    # The DETAILED_SUBQUESTION_ANSWERING_PROMPT in Prompt/answer.py seems to be a template 
    # but it doesn't have {{passages}} or {{subquestion}} placeholders in the text I read earlier?
    # Let's re-read Prompt/answer.py content carefully.
    # Ah, I see it has {{main_query}} and "---Previous Context---".
    # But where does the current passage and subquestion go?
    # The prompt text I read earlier ends with "---Previous Context (IMPORTANT - May contain the answer!)---".
    # It seems I need to append the context, passages, and the question manually.
    
    full_prompt = prompt + "\n" + previous_context + "\n\n---Current Information---\n" + passage_text + "\n\n---Sub-Question---\n" + actual_question + "\n\n---Answer---\n"
    
    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise question answering system."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.1,
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

async def process_question(client, retriever, item):
    question = item['question']
    
    # 1. Decompose
    decomposition_result = await decompose_query(client, question)
    if not decomposition_result or not decomposition_result.get('success'):
        return {'question': question, 'error': 'Decomposition failed'}
    
    decomposition = decomposition_result['decomposition']
        
    all_passages = {} # title -> text
    
    # 2. Process Sub-questions
    for sq in decomposition.subquestions:
        # Build context
        prev_context = ""
        if sq.depends_on:
            for dep_id in sq.depends_on:
                dep_sq = decomposition.get_subquestion(dep_id)
                if dep_sq and dep_sq.answer:
                    prev_context += f"Q: {dep_sq.question}\nA: {dep_sq.answer}\n\n"
        
        answer, results = await answer_subquestion(client, retriever, sq, decomposition, prev_context)
        sq.answer = answer
        sq.retrieved_passages = results
        
        # Collect passages
        for res in results:
            all_passages[res['title']] = res['text']
            
    # 3. Final Answer
    # Format all unique passages
    unique_passages_text = ""
    for i, (title, text) in enumerate(all_passages.items(), 1):
        unique_passages_text += f"[{i}] {title}\n{text}\n\n"
        
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
        temperature=0.1,
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
    
    return {
        'id': item.get('_id'),
        'question': question,
        'gold_answer': item['answer'],
        'predicted_answer': final_answer,
        'answer_aliases': item.get('answer_aliases', []),
        'decomposition': [sq.to_dict() for sq in decomposition.subquestions]
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['hotpotqa', 'musique'])
    args = parser.parse_args()
    
    load_dotenv()
    init_logger()
    
    # Config
    if args.dataset == 'hotpotqa':
        data_path = 'HotpotQA/hotpotqa_sample_200.json'
        cache_path = 'HotpotQA/passage_embeddings_sample_200.npz'
    else:
        data_path = 'MuSiQue/musique_sample_200.json'
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
    print(f"Processing {len(data)} questions with QD...")
    
    import time
    start_time = time.time()
    
    # Process in batches for concurrency
    batch_size = 50  # Adjust based on rate limits
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        tasks = [process_question(chat_client, retriever, item) for item in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        print(f"Processed {min(i+batch_size, len(data))}/{len(data)}")
            
    total_time = time.time() - start_time
    
    # Save Results
    output_file = f'Results/NaiveRAG/NaiveRAG_passage_QD_{args.dataset}.json'
    os.makedirs('Results/NaiveRAG', exist_ok=True)
    
    output = {
        'config': {
            'pipeline': 'NaiveRAG_QD',
            'dataset': args.dataset,
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

if __name__ == "__main__":
    asyncio.run(main())
