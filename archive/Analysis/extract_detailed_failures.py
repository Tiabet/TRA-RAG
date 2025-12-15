import json
import os

def extract_detailed_failures():
    failures_path = 'Analysis/recoverable_recall_failures.json'
    results_path = 'Analysis/recoverable_musique_results.json'
    output_path = 'Analysis/recoverable_recall_failures_detailed.json'

    print(f"Loading failure IDs from {failures_path}...")
    with open(failures_path, 'r', encoding='utf-8') as f:
        failures_data = json.load(f)
    
    failure_ids = set(item['id'] for item in failures_data)
    print(f"Found {len(failure_ids)} failure cases.")

    print(f"Loading full results from {results_path}...")
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    detailed_failures = []
    
    for res in results_data:
        # Handle ID key difference if any
        q_id = res.get('id') or res.get('_id')
        
        if q_id in failure_ids:
            simplified_res = {
                'id': q_id,
                'question': res.get('question'),
                'gold_answer': res.get('gold_answer'),
                'predicted_answer': res.get('predicted_answer'),
                'decomposition': {
                    'main_query': res.get('decomposition', {}).get('main_query'),
                    'subquestions': []
                }
            }

            if 'decomposition' in res and 'subquestions' in res['decomposition']:
                for sq in res['decomposition']['subquestions']:
                    simplified_sq = {
                        'id': sq.get('id'),
                        'question': sq.get('question'),
                        'answer': sq.get('answer'),
                        'depends_on': sq.get('depends_on', []),
                        'retrieved_passages': []
                    }

                    if 'retrieved_passages' in sq:
                        for p in sq['retrieved_passages']:
                            simplified_p = {
                                'title': p.get('title'),
                                'matched_path': p.get('matched_path'),
                                'matched_value': p.get('matched_value'),
                                'score': p.get('score'),
                                'bm25_score': p.get('bm25_score'),
                                'dense_score': p.get('dense_score')
                            }
                            simplified_sq['retrieved_passages'].append(simplified_p)
                    
                    simplified_res['decomposition']['subquestions'].append(simplified_sq)
            
            detailed_failures.append(simplified_res)

    print(f"Extracted {len(detailed_failures)} detailed failure reports.")
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_failures, f, indent=2, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    extract_detailed_failures()
