"""
Verify Original Analysis - Check if old predictions were really all wrong
"""
import json

# Load results
with open('improved_prompt_test_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

cases = results['results']

# Check old predictions
old_correct_count = 0
old_correct_cases = []

for c in cases:
    gold = c['gold_answer'].lower().strip()
    old_pred = c['old_prediction'].lower().strip()
    
    # Use original analysis logic: gold in pred
    if gold in old_pred:
        old_correct_count += 1
        old_correct_cases.append({
            'question': c['question'],
            'gold': c['gold_answer'],
            'old_pred': c['old_prediction']
        })

print(f"Total cases: {len(cases)}")
print(f"Old predictions where gold IN pred: {old_correct_count}/{len(cases)}")
print(f"\nThis should be 0 if all 62 cases were originally 'wrong'")

if old_correct_count > 0:
    print(f"\n⚠️  WARNING: {old_correct_count} cases were actually correct in original analysis!")
    print("\nExamples:")
    for i, case in enumerate(old_correct_cases[:5], 1):
        print(f"\n{i}. {case['question'][:80]}...")
        print(f"   Gold: {case['gold']}")
        print(f"   Old:  {case['old_pred']}")
