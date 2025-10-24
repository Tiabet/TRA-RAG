"""
Extract Still Wrong Cases from Improved Prompt Test
====================================================
개선된 프롬프트로도 여전히 틀린 케이스들을 추출합니다.
"""

import json
from pathlib import Path

def extract_still_wrong_cases():
    """Extract cases that are still wrong after prompt improvement"""
    
    # Load test results
    results_path = Path('improved_prompt_test_results.json')
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return
    
    with results_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter still wrong cases
    all_results = data.get('results', [])
    still_wrong = [r for r in all_results if r['status'] == "❌ STILL WRONG"]
    
    print(f"Total cases tested: {len(all_results)}")
    print(f"Still wrong: {len(still_wrong)}")
    
    # Prepare detailed analysis
    still_wrong_detailed = []
    
    for case in still_wrong:
        # Analyze the error pattern
        gold = case['gold_answer'].lower()
        old_pred = case['old_prediction'].lower()
        new_pred = case['new_prediction'].lower()
        
        # Check if it's format mismatch or completely wrong
        gold_words = set(gold.split())
        new_pred_words = set(new_pred.split())
        overlap = gold_words & new_pred_words
        
        if len(overlap) > 0 and len(overlap) >= len(gold_words) * 0.5:
            error_type = "Format Mismatch (>50% word overlap)"
        elif len(overlap) > 0:
            error_type = "Partial Match (some word overlap)"
        else:
            error_type = "Completely Wrong (no word overlap)"
        
        still_wrong_detailed.append({
            'question_id': case['qid'],
            'question': case['question'],
            'gold_answer': case['gold_answer'],
            'old_prediction': case['old_prediction'],
            'new_prediction': case['new_prediction'],
            'error_type': error_type,
            'word_overlap': len(overlap),
            'gold_words': len(gold_words)
        })
    
    # Save to file
    output_path = Path('still_wrong_cases.json')
    with output_path.open('w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_still_wrong': len(still_wrong),
                'by_error_type': {
                    'format_mismatch': len([c for c in still_wrong_detailed if 'Format Mismatch' in c['error_type']]),
                    'partial_match': len([c for c in still_wrong_detailed if 'Partial Match' in c['error_type']]),
                    'completely_wrong': len([c for c in still_wrong_detailed if 'Completely Wrong' in c['error_type']])
                }
            },
            'cases': still_wrong_detailed
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(still_wrong)} still wrong cases to: {output_path}")
    
    # Print summary by error type
    print("\n" + "="*100)
    print("ERROR TYPE BREAKDOWN")
    print("="*100)
    
    format_mismatch = [c for c in still_wrong_detailed if 'Format Mismatch' in c['error_type']]
    partial_match = [c for c in still_wrong_detailed if 'Partial Match' in c['error_type']]
    completely_wrong = [c for c in still_wrong_detailed if 'Completely Wrong' in c['error_type']]
    
    print(f"Format Mismatch (>50% overlap): {len(format_mismatch)}")
    print(f"Partial Match (some overlap): {len(partial_match)}")
    print(f"Completely Wrong (no overlap): {len(completely_wrong)}")
    
    # Show examples of each type
    print("\n" + "="*100)
    print("EXAMPLES BY ERROR TYPE")
    print("="*100)
    
    if format_mismatch:
        print("\n[1] Format Mismatch Example:")
        ex = format_mismatch[0]
        print(f"  Question: {ex['question']}")
        print(f"  Gold: {ex['gold_answer']}")
        print(f"  Pred: {ex['new_prediction']}")
    
    if partial_match:
        print("\n[2] Partial Match Example:")
        ex = partial_match[0]
        print(f"  Question: {ex['question']}")
        print(f"  Gold: {ex['gold_answer']}")
        print(f"  Pred: {ex['new_prediction']}")
    
    if completely_wrong:
        print("\n[3] Completely Wrong Example:")
        ex = completely_wrong[0]
        print(f"  Question: {ex['question']}")
        print(f"  Gold: {ex['gold_answer']}")
        print(f"  Old: {ex['old_prediction']}")
        print(f"  New: {ex['new_prediction']}")
    
    return still_wrong_detailed


if __name__ == "__main__":
    print("="*100)
    print("Extracting Still Wrong Cases")
    print("="*100)
    extract_still_wrong_cases()
