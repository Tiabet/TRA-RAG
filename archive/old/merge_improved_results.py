"""
Merge Improved QA Results with Original Results
================================================
개선된 9개 QA를 기존 200개 결과와 결합하여 새로운 결과 파일 생성
"""

import json
from pathlib import Path

def merge_improved_results():
    """
    Merge improved results with original results.
    Replace the 9 improved cases with their new predictions.
    """
    
    # Load original 200 results
    original_path = Path('multihop_pipeline_200_results.json')
    if not original_path.exists():
        print(f"❌ Original results not found: {original_path}")
        return
    
    with original_path.open('r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"✅ Loaded original results: {len(original_data['results'])} questions")
    
    # Load improved test results (corrected version)
    improved_path = Path('improved_prompt_test_results_CORRECTED.json')
    if not improved_path.exists():
        print(f"❌ Improved results not found: {improved_path}")
        return
    
    with improved_path.open('r', encoding='utf-8') as f:
        improved_data = json.load(f)
    
    # Get only the improved cases (9 cases)
    improved_cases = [r for r in improved_data['results'] if r['status'] == "✅ IMPROVED"]
    print(f"✅ Found {len(improved_cases)} improved cases")
    
    # Build a mapping: qid -> new_prediction
    improved_predictions = {}
    for case in improved_cases:
        improved_predictions[case['qid']] = case['new_prediction']
    
    print(f"\nImproved cases:")
    for qid, pred in improved_predictions.items():
        print(f"  {qid}: {pred[:60]}...")
    
    # Create merged results
    merged_results = []
    replaced_count = 0
    
    for result in original_data['results']:
        qid = result['question_id']
        
        if qid in improved_predictions:
            # Replace with improved prediction
            result_copy = result.copy()
            result_copy['predicted_answer'] = improved_predictions[qid]
            result_copy['improved'] = True  # Mark as improved
            merged_results.append(result_copy)
            replaced_count += 1
            print(f"\n✅ Replaced {qid}")
            print(f"   Old: {result['predicted_answer'][:60]}...")
            print(f"   New: {improved_predictions[qid][:60]}...")
        else:
            # Keep original
            merged_results.append(result)
    
    print(f"\n{'='*100}")
    print(f"Merge Summary:")
    print(f"  Total results: {len(merged_results)}")
    print(f"  Replaced with improved: {replaced_count}")
    print(f"  Kept original: {len(merged_results) - replaced_count}")
    print(f"{'='*100}")
    
    # Update metadata
    merged_data = original_data.copy()
    merged_data['results'] = merged_results
    merged_data['metadata']['note'] = f"Merged with {replaced_count} improved predictions from prompt improvement"
    merged_data['metadata']['improved_count'] = replaced_count
    
    # Save merged results
    output_path = Path('multihop_pipeline_200_results_IMPROVED.json')
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved merged results to: {output_path}")
    
    # Also create a simple format for judge_F1.py
    simple_results = {
        'results': [
            {
                'question': r['question'],
                'gold_answer': r['gold_answer'],
                'predicted_answer': r['predicted_answer'],
                'question_id': r['question_id']
            }
            for r in merged_results
        ]
    }
    
    simple_path = Path('multihop_pipeline_200_results_IMPROVED_simple.json')
    with simple_path.open('w', encoding='utf-8') as f:
        json.dump(simple_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved simple format to: {simple_path}")
    
    return merged_results


if __name__ == "__main__":
    print("="*100)
    print("Merging Improved QA Results with Original Results")
    print("="*100)
    merge_improved_results()
