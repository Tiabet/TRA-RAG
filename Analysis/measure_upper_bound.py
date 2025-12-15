import asyncio
import json
import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Prompt.subquestion_answering_prompt import FINAL_ANSWER_SYNTHESIS_PROMPT

# Load environment variables
load_dotenv()

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_gold_passages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract gold passages from the item.
    Returns a list of dicts with 'title' and 'original_passage'.
    """
    supporting_facts = item.get('supporting_facts', [])
    # supporting_facts is list of [title, sent_idx]
    # We want unique titles
    gold_titles = set(fact[0] for fact in supporting_facts)
    
    context = item.get('context', [])
    # context is list of [title, [sentences]]
    
    passages = []
    seen_titles = set()
    
    for ctx in context:
        title = ctx[0]
        if title in gold_titles and title not in seen_titles:
            sentences = ctx[1]
            text = ' '.join(sentences)
            passages.append({
                'title': title,
                'original_passage': text
            })
            seen_titles.add(title)
            
    return passages

async def generate_answer(client, question, passages):
    # Format passages
    passage_texts = []
    for i, p in enumerate(passages, 1):
        title = p['title']
        text = p['original_passage']
        passage_texts.append(f"[{i}] {title}\n{text}")
    
    passages_text = '\n\n'.join(passage_texts) if passage_texts else "No passages."
    
    # Prepare Prompt
    # We provide a dummy chain since we are skipping decomposition
    subquestion_chain = "N/A (Gold Context Provided directly)"
    
    prompt = FINAL_ANSWER_SYNTHESIS_PROMPT.replace("{{main_question}}", question)
    prompt = prompt.replace("{{subquestion_chain}}", subquestion_chain)
    prompt = prompt.replace("{{passages}}", passages_text)
    
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise question answering system. Give short, direct answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        answer = response.choices[0].message.content.strip()
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error"

async def process_item(client, item):
    question = item['question']
    gold_answer = item['answer']
    qid = item.get('_id', 'unknown')
    
    passages = get_gold_passages(item)
    predicted_answer = await generate_answer(client, question, passages)
    
    return {
        'question_id': qid,
        'question': question,
        'gold_answer': gold_answer,
        'predicted_answer': predicted_answer,
        'num_gold_passages': len(passages)
    }

async def run_dataset(client, dataset_path, output_filename):
    print(f"\nProcessing {output_filename}...")
    data = load_json(dataset_path)
    
    # Limit for testing if needed, but user wants full run (implied)
    # data = data[:5] 
    
    tasks = []
    sem = asyncio.Semaphore(20) # Concurrency limit
    
    async def run_with_sem(item):
        async with sem:
            return await process_item(client, item)
    
    for item in data:
        tasks.append(run_with_sem(item))
        
    results = await asyncio.gather(*tasks)
    
    # Save results
    base_dir = r"c:\Development\ChunkRAG_v2"
    output_path = os.path.join(base_dir, "Results", output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(results)} results to {output_path}")

async def main():
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    base_dir = r"c:\Development\ChunkRAG_v2"
    
    # Run MuSiQue
    musique_path = os.path.join(base_dir, "MuSiQue", "musique_sample_200.json")
    if os.path.exists(musique_path):
        await run_dataset(client, musique_path, "upper_bound_musique.json")
        
    # Run HotpotQA
    hotpot_path = os.path.join(base_dir, "HotpotQA", "hotpotqa_sample_200.json")
    if os.path.exists(hotpot_path):
        await run_dataset(client, hotpot_path, "upper_bound_hotpotqa.json")

if __name__ == "__main__":
    asyncio.run(main())
