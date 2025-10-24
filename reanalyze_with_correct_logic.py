"""
Re-analyze Improved Prompt Test Results with Correct Logic
===========================================================
Use SAME logic as original analysis: gold in pred (not bidirectional)
"""
import json
from pathlib import Path

# Load results
with open('improved_prompt_test_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

cases = data['results']

# Re-analyze with CORRECT logic
improvements = 0
still_wrong = 0
new_errors = 0
already_correct = 0  # This should be 0!

comparison_results = []

for c in cases:
    gold = c['gold_answer'].lower().strip()
    old_pred = c['old_prediction'].lower().strip()
    new_pred = c['new_prediction'].lower().strip()
    
    # Use ORIGINAL ANALYSIS LOGIC: gold in pred (one direction only!)
    old_correct = gold in old_pred if (gold and old_pred and old_pred != 'n/a') else False
    new_correct = gold in new_pred if (gold and new_pred and new_pred != 'n/a') else False
    
    # Categorize
    if not old_correct and new_correct:
        improvements += 1
        status = "✅ IMPROVED"
    elif not old_correct and not new_correct:
        still_wrong += 1
        status = "❌ STILL WRONG"
    elif old_correct and not new_correct:
        new_errors += 1
        status = "⚠️  NEW ERROR"
    else:  # old_correct and new_correct
        already_correct += 1
        status = "✓ Already correct"
    
    comparison_results.append({
        'qid': c['qid'],
        'question': c['question'],
        'gold_answer': c['gold_answer'],
        'old_prediction': c['old_prediction'],
        'new_prediction': c['new_prediction'],
        'status': status,
        'old_correct': old_correct,
        'new_correct': new_correct
    })

# Summary
total = len(cases)
print("="*100)
print("CORRECTED IMPROVEMENT ANALYSIS")
print("="*100)
print(f"\nTotal Cases Retested: {total}")
print(f"  ✅ Improved (Wrong → Correct): {improvements} ({improvements/total*100:.1f}%)")
print(f"  ❌ Still Wrong: {still_wrong} ({still_wrong/total*100:.1f}%)")
print(f"  ⚠️  New Errors: {new_errors} ({new_errors/total*100:.1f}%)")
print(f"  ✓ Already Correct: {already_correct} ({already_correct/total*100:.1f}%) [Should be 0!]")

improvement_rate = improvements / total * 100 if total > 0 else 0
print(f"\n🎯 Improvement Rate: {improvement_rate:.1f}%")

# Show improved cases
improved_cases = [r for r in comparison_results if r['status'] == "✅ IMPROVED"]

if improved_cases:
    print(f"\n{'='*100}")
    print(f"IMPROVED CASES ({len(improved_cases)} total)")
    print(f"{'='*100}")
    
    for i, case in enumerate(improved_cases[:10], 1):
        print(f"\n{i}. {case['question'][:80]}...")
        print(f"   Gold: {case['gold_answer']}")
        print(f"   Old:  {case['old_prediction']} ❌")
        print(f"   New:  {case['new_prediction']} ✅")

# Show still wrong cases
if still_wrong > 0:
    wrong_cases = [r for r in comparison_results if r['status'] == "❌ STILL WRONG"]
    
    print(f"\n{'='*100}")
    print(f"STILL WRONG CASES ({len(wrong_cases)} total) - First 5")
    print(f"{'='*100}")
    
    for i, case in enumerate(wrong_cases[:5], 1):
        print(f"\n{i}. {case['question'][:80]}...")
        print(f"   Gold: {case['gold_answer']}")
        print(f"   Old:  {case['old_prediction']} ❌")
        print(f"   New:  {case['new_prediction']} ❌")

# Save corrected results
output = {
    'summary': {
        'total_cases': total,
        'improvements': improvements,
        'still_wrong': still_wrong,
        'new_errors': new_errors,
        'already_correct': already_correct,
        'improvement_rate': improvement_rate
    },
    'results': comparison_results
}

output_path = Path('improved_prompt_test_results_CORRECTED.json')
with output_path.open('w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n{'='*100}")
print(f"✅ Corrected results saved to: {output_path}")
print(f"{'='*100}")
