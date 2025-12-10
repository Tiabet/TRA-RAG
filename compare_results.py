import json
import os

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_question(q):
    return q.strip().lower()

def get_retrieved_titles(result_item):
    titles = set()
    if 'decomposition' in result_item and result_item['decomposition']:
        # Check if decomposition is a dict (new format) or list
        decomp = result_item['decomposition']
        if isinstance(decomp, dict):
            # New format: main_query, subquestions
            if 'subquestions' in decomp:
                for sq in decomp['subquestions']:
                    if 'retrieved_passages' in sq:
                        for p in sq['retrieved_passages']:
                            titles.add(p['title'])
        elif isinstance(decomp, list):
            # Old format or list of subquestions
            for sq in decomp:
                if 'retrieved_passages' in sq:
                    for p in sq['retrieved_passages']:
                        titles.add(p['title'])
    
    # Also check top-level retrieved_passages if they exist
    if 'retrieved_passages' in result_item:
        for p in result_item['retrieved_passages']:
            titles.add(p['title'])
            
    return titles

def main():
    # Paths
    upper_bound_path = 'Results/upper_bound_original_evaluation.json'
    my_eval_path = 'Results/llm_eval_test_hotpot_v3_200_results.json'
    my_results_path = 'Results/test_hotpot_v3_200_results.json'
    gold_data_path = 'HotpotQA/hotpotqa_sample_200.json'

    # Load data
    print("Loading data...")
    upper_bound_data = load_json(upper_bound_path)
    my_eval_data = load_json(my_eval_path)
    my_results_data = load_json(my_results_path)
    gold_data = load_json(gold_data_path)

    # Indexing
    upper_map = {normalize_question(item['question']): item for item in upper_bound_data['evaluations']}
    my_eval_map = {normalize_question(item['question']): item for item in my_eval_data['results']}
    my_results_map = {normalize_question(item['question']): item for item in my_results_data['results']}
    gold_map = {normalize_question(item['question']): item for item in gold_data}

    # Categories
    both_correct = []
    both_incorrect = []
    upper_correct_my_incorrect = []
    upper_incorrect_my_correct = []

    # Analysis
    print("Analyzing results...")
    for q_norm, my_eval in my_eval_map.items():
        if q_norm not in upper_map:
            continue
        
        upper_eval = upper_map[q_norm]
        
        my_verdict = my_eval['evaluation']['verdict'] == 'CORRECT'
        upper_verdict = upper_eval['verdict'] == 'CORRECT'

        item_info = {
            'question': my_eval['question'],
            'gold_answer': my_eval['gold_answer'],
            'my_answer': my_eval['predicted_answer'],
            'upper_answer': upper_eval['predicted_answer'],
            'my_reason': my_eval['evaluation']['reason'],
            'upper_reason': upper_eval['reason']
        }

        if my_verdict and upper_verdict:
            both_correct.append(item_info)
        elif not my_verdict and not upper_verdict:
            both_incorrect.append(item_info)
        elif upper_verdict and not my_verdict:
            # Analyze retrieval for this case
            if q_norm in my_results_map and q_norm in gold_map:
                retrieved_titles = get_retrieved_titles(my_results_map[q_norm])
                gold_titles = set(fact[0] for fact in gold_map[q_norm]['supporting_facts'])
                
                missing_titles = gold_titles - retrieved_titles
                item_info['missing_passages'] = list(missing_titles)
                item_info['retrieved_titles'] = list(retrieved_titles)
                item_info['gold_titles'] = list(gold_titles)
            
            upper_correct_my_incorrect.append(item_info)
        elif not upper_verdict and my_verdict:
            upper_incorrect_my_correct.append(item_info)

    # Output Report
    output_data = {
        "regressions": [],
        "improvements": []
    }

    # Helper to build case object
    def build_case(item_info, q_norm):
        case = {
            "question": item_info['question'],
            "gold_answer": item_info['gold_answer'],
            "my_answer": item_info['my_answer'],
            "upper_bound_answer": item_info['upper_answer'],
            "my_reason": item_info['my_reason'],
            "upper_reason": item_info['upper_reason']
        }
        
        # Add retrieval info if available
        if q_norm in my_results_map and q_norm in gold_map:
            retrieved_titles = get_retrieved_titles(my_results_map[q_norm])
            gold_titles = set(fact[0] for fact in gold_map[q_norm]['supporting_facts'])
            missing_titles = gold_titles - retrieved_titles
            
            case['supporting_fact_titles'] = list(gold_titles)
            case['retrieved_titles'] = list(retrieved_titles)
            case['missing_titles'] = list(missing_titles)
            
            if len(gold_titles) > 0:
                case['recall'] = 1.0 - (len(missing_titles) / len(gold_titles))
            else:
                case['recall'] = 0.0 # Should not happen usually
        
        return case

    print("Analyzing results...")
    for q_norm, my_eval in my_eval_map.items():
        if q_norm not in upper_map:
            continue
        
        upper_eval = upper_map[q_norm]
        
        my_verdict = my_eval['evaluation']['verdict'] == 'CORRECT'
        upper_verdict = upper_eval['verdict'] == 'CORRECT'

        item_info = {
            'question': my_eval['question'],
            'gold_answer': my_eval['gold_answer'],
            'my_answer': my_eval['predicted_answer'],
            'upper_answer': upper_eval['predicted_answer'],
            'my_reason': my_eval['evaluation']['reason'],
            'upper_reason': upper_eval['reason']
        }

        if upper_verdict and not my_verdict:
            # Regression
            output_data["regressions"].append(build_case(item_info, q_norm))
        elif not upper_verdict and my_verdict:
            # Improvement
            output_data["improvements"].append(build_case(item_info, q_norm))

    # Save to JSON
    output_path = 'Results/comparison_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    main()
