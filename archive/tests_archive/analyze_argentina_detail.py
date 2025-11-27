import json

with open('multihop_pipeline_200_checkpoint.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Argentina case
argentina = [r for r in data['results'] if 'Argentina' in r['question']][0]

print("="*80)
print("Argentina 케이스 상세 분석")
print("="*80)

print(f"\nQuestion: {argentina['question']}")
print(f"Gold Answer: {argentina['gold_answer']}")
print(f"Predicted: {argentina['predicted_answer']}")

print("\n" + "="*80)
print("Sub-Questions:")
print("="*80)

for sq in argentina['decomposition']['subquestions']:
    print(f"\n{sq['id']}: {sq['question']}")
    print(f"  Answer: {sq['answer']}")
    print(f"  Retrieved: {sq.get('retrieved_passages', [])}")

print("\n" + "="*80)
print("Retrieved Passages (Total):")
print("="*80)

for title in argentina['retrieved_passages']['titles']:
    print(f"  - {title}")

print("\n" + "="*80)
print("SQ별 Retrieved Passages:")
print("="*80)

for sq_ret in argentina['retrieved_passages']['by_subquestion']:
    print(f"\n{sq_ret['subquestion_id']}: {sq_ret['titles']}")
