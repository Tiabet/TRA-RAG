import json
import os

def extract_recall_failures():
    qa_path = 'Analysis/recoverable_musique_qa.json'
    results_path = 'Analysis/recoverable_musique_results.json'
    output_path = 'Analysis/recoverable_recall_failures.json'

    print(f"Loading QA data from {qa_path}...")
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    # Map ID to gold titles
    gold_map = {}
    for item in qa_data:
        # supporting_facts is a list of [title, sent_idx]
        # We just need the unique titles
        titles = set(fact[0] for fact in item['supporting_facts'])
        gold_map[item['_id']] = titles

    print(f"Loading results from {results_path}...")
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    failures = []
    
    for res in results_data:
        # Handle ID key difference if any (results usually use 'id', qa uses '_id')
        q_id = res.get('id') or res.get('_id')
        
        if q_id not in gold_map:
            print(f"Warning: ID {q_id} not found in QA data.")
            continue

        gold_titles = gold_map[q_id]
        
        # Collect retrieved titles
        retrieved_titles = set()
        
        # Check if decomposition exists
        if 'decomposition' in res and 'subquestions' in res['decomposition']:
            for sq in res['decomposition']['subquestions']:
                if 'retrieved_passages' in sq:
                    for p in sq['retrieved_passages']:
                        retrieved_titles.add(p['title'])
        
        # Also check top-level retrieved_passages if they exist (fallback or non-decomp mode)
        if 'retrieved_passages' in res:
             for p in res['retrieved_passages']:
                retrieved_titles.add(p['title'])

        # Check for missing titles
        missing_titles = gold_titles - retrieved_titles
        
        if missing_titles:
            failures.append({
                'id': q_id,
                'question': res.get('question', ''),
                'gold_titles': list(gold_titles),
                'retrieved_titles': list(retrieved_titles),
                'missing_titles': list(missing_titles)
            })

    print(f"Total Recoverable Questions: {len(results_data)}")
    print(f"Total Recall Failures: {len(failures)}")
    print(f"Failure Rate: {len(failures)/len(results_data)*100:.2f}%")

    print(f"Saving failures to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    extract_recall_failures()
