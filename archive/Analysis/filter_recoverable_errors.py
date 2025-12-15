import json
import os

def normalize_question(q):
    return q.strip().lower() if q else ""

def main():
    # File Paths
    current_results_path = r"Results/llm_eval_test_musique_v4_200_results.json"
    upper_bound_results_path = r"Results/llm_eval_upper_bound_musique.json"
    original_qa_path = r"MuSiQue/musique_sample_200.json"
    output_qa_path = r"Analysis/recoverable_musique_qa.json"

    # Load Data
    print(f"Loading Current Results from {current_results_path}...")
    with open(current_results_path, 'r', encoding='utf-8') as f:
        current_results = json.load(f)

    print(f"Loading Upper Bound Results from {upper_bound_results_path}...")
    with open(upper_bound_results_path, 'r', encoding='utf-8') as f:
        upper_bound_results = json.load(f)

    print(f"Loading Original QA Data from {original_qa_path}...")
    with open(original_qa_path, 'r', encoding='utf-8') as f:
        original_qa = json.load(f)

    # Create Maps
    # Map Question Text -> Verdict
    current_verdicts = {}
    for item in current_results['results']:
        q_text = normalize_question(item.get('question'))
        verdict = item.get('evaluation', {}).get('verdict')
        if q_text:
            current_verdicts[q_text] = verdict

    upper_bound_verdicts = {}
    for item in upper_bound_results['results']:
        q_text = normalize_question(item.get('question'))
        verdict = item.get('evaluation', {}).get('verdict')
        if q_text:
            upper_bound_verdicts[q_text] = verdict

    # Identify Recoverable Errors
    recoverable_questions = []
    recoverable_count = 0
    
    print("\nAnalyzing...")
    for item in original_qa:
        q_text = normalize_question(item.get('question'))
        
        current_verdict = current_verdicts.get(q_text)
        upper_bound_verdict = upper_bound_verdicts.get(q_text)

        if current_verdict == 'INCORRECT' and upper_bound_verdict == 'CORRECT':
            recoverable_questions.append(item)
            recoverable_count += 1
        
        # Debug print for first few mismatches
        # if current_verdict == 'INCORRECT' and upper_bound_verdict == 'INCORRECT':
        #     print(f"Both Incorrect: {q_text[:50]}...")

    print(f"\nFound {recoverable_count} recoverable errors (Current=INCORRECT, UpperBound=CORRECT).")
    
    # Save Filtered QA
    print(f"Saving filtered QA data to {output_qa_path}...")
    with open(output_qa_path, 'w', encoding='utf-8') as f:
        json.dump(recoverable_questions, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
