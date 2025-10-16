"""
Extract entities from questions in a dataset using the entity extraction prompt.
"""
import os
import json
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

# Load environment variables
load_dotenv()

# Import the entity extraction prompt
from Prompt.entity_extraction_prompt import ENTITY_EXTRACTION_PROMPT

def parse_args():
    parser = argparse.ArgumentParser(description='Extract entities from dataset questions')
    parser.add_argument('-i', '--input', required=True, help='Input dataset file path')
    parser.add_argument('-o', '--output', help='Output file path (default: input_file_entities.json)')
    parser.add_argument('--concurrency', type=int, default=20, help='Number of concurrent requests (default: 20)')
    parser.add_argument('--max-questions', type=int, help='Maximum number of questions to process')
    return parser.parse_args()

def initialize_llm_client():
    """Initialize the AsyncOpenAI client with Alice API settings"""
    api_key = os.getenv('ALICE_OPENAI_KEY')
    base_url = os.getenv('ALICE_CHAT_URL')
    
    if not api_key or not base_url:
        raise ValueError("ALICE_OPENAI_KEY and ALICE_CHAT_URL must be set in .env file")
    
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )

async def extract_entities(client, question, semaphore):
    """Extract entities from a single question using LLM"""
    async with semaphore:
        try:
            # Format the prompt with the question
            formatted_prompt = ENTITY_EXTRACTION_PROMPT.replace("__QUESTION__", question)
            
            response = await client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Remove code block markers if present
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            # Parse JSON
            result = json.loads(result_text)
            
            return {
                'success': True,
                'entities': result.get('entities', [])
            }
            
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f'JSON parse error: {str(e)}',
                'raw_response': result_text if 'result_text' in locals() else None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

async def process_dataset(client, dataset, max_questions, concurrency):
    """Process all questions in the dataset"""
    # Limit number of questions if specified
    questions_to_process = dataset[:max_questions] if max_questions else dataset
    
    print(f"\nProcessing {len(questions_to_process)} questions...")
    
    semaphore = asyncio.Semaphore(concurrency)
    
    # Create tasks for all questions
    tasks = [extract_entities(client, item['question'], semaphore) for item in questions_to_process]
    
    # Process with gather to maintain order, wrap with tqdm for progress
    results = []
    with tqdm(total=len(tasks), desc="Extracting entities") as pbar:
        # Process in batches for progress updates
        batch_size = concurrency
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            pbar.update(len(batch))
    
    # Combine results with original data
    output_data = []
    success_count = 0
    fail_count = 0
    
    for i, item in enumerate(questions_to_process):
        result = results[i]
        output_item = {
            '_id': item.get('_id'),
            'question': item['question'],
            'type': item.get('type'),
            'level': item.get('level'),
            'extraction_result': result
        }
        output_data.append(output_item)
        
        if result['success']:
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Success: {success_count}/{len(questions_to_process)} ({success_count/len(questions_to_process)*100:.1f}%)")
    print(f"Failed: {fail_count}/{len(questions_to_process)} ({fail_count/len(questions_to_process)*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return output_data

def main():
    args = parse_args()
    
    # Generate output path if not specified
    if args.output:
        output_path = args.output
    else:
        input_base = os.path.splitext(args.input)[0]
        output_path = f"{input_base}_entities.json"
    
    print(f"Input file: {args.input}")
    print(f"Output file: {output_path}")
    print(f"Concurrency: {args.concurrency}")
    if args.max_questions:
        print(f"Max questions: {args.max_questions}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        if args.input.endswith('.jsonl'):
            dataset = [json.loads(line) for line in f]
        else:
            dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} questions")
    
    # Initialize client
    client = initialize_llm_client()
    
    # Process dataset
    output_data = asyncio.run(process_dataset(
        client, 
        dataset, 
        args.max_questions,
        args.concurrency
    ))
    
    # Save results
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved successfully!")
    
    # Print some example results
    print("\n" + "="*60)
    print("Sample results:")
    print("="*60)
    for i, item in enumerate(output_data[:3]):
        print(f"\n{i+1}. Question: {item['question']}")
        print(f"   Type: {item['type']}")
        if item['extraction_result']['success']:
            entities = item['extraction_result']['entities']
            print(f"   Entities ({len(entities)}):")
            for entity in entities:
                print(f"     - {entity['entity_name']} ({entity['type']}/{entity.get('subtype', 'N/A')})")
        else:
            print(f"   Error: {item['extraction_result']['error']}")

if __name__ == "__main__":
    main()
