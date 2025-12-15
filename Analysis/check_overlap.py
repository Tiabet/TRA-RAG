import json

def normalize_question(q):
    return q.strip().lower() if q else ""

def main():
    # File Paths
    current_results_path = r"Results/llm_eval_test_musique_v4_200_results.json"
    upper_bound_results_path = r"Results/llm_eval_upper_bound_musique.json"

    # Load Data
    with open(current_results_path, 'r', encoding='utf-8') as f:
        current_results = json.load(f)
    with open(upper_bound_results_path, 'r', encoding='utf-8') as f:
        upper_bound_results = json.load(f)

    # Create Sets of Question Texts
    current_correct = set()
    for item in current_results['results']:
        if item.get('evaluation', {}).get('verdict') == 'CORRECT':
            current_correct.add(normalize_question(item.get('question')))

    upper_bound_correct = set()
    for item in upper_bound_results['results']:
        if item.get('evaluation', {}).get('verdict') == 'CORRECT':
            upper_bound_correct.add(normalize_question(item.get('question')))

    # Calculate Sets
    intersection = current_correct.intersection(upper_bound_correct)
    recoverable = upper_bound_correct - current_correct
    reverse_case = current_correct - upper_bound_correct

    print(f"Current System Correct: {len(current_correct)}")
    print(f"Upper Bound Correct: {len(upper_bound_correct)}")
    print(f"Intersection (Both Correct): {len(intersection)}")
    print("-" * 30)
    print(f"Recoverable (Upper Correct, Current Incorrect): {len(recoverable)}")
    print(f"Reverse Case (Current Correct, Upper Incorrect): {len(reverse_case)}")
    print("-" * 30)
    print(f"Math Check: {len(upper_bound_correct)} (Upper) - {len(intersection)} (Shared) = {len(recoverable)}")
    print(f"Math Check: {len(current_correct)} (Current) - {len(intersection)} (Shared) = {len(reverse_case)}")

if __name__ == "__main__":
    main()
