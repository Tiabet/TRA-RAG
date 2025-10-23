import json

# Load checkpoint
with open('multihop_pipeline_200_checkpoint.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results'][:120]

# Count insufficient
insufficient = [r for r in results if 'Insufficient' in r['predicted_answer']]
exact_match = [r for r in results if r['exact_match']]

print(f"총 120개 중:")
print(f"- Insufficient information: {len(insufficient)}개 ({len(insufficient)/120*100:.1f}%)")
print(f"- Exact Match 성공: {len(exact_match)}개 ({len(exact_match)/120*100:.1f}%)")
print(f"- Token F1 > 0: {len([r for r in results if r['token_f1'] > 0])}개")

# Analyze retrieval
no_passages = [r for r in insufficient if r['retrieved_passages']['count'] == 0]
has_passages = [r for r in insufficient if r['retrieved_passages']['count'] > 0]

print(f"\n검색 분석:")
print(f"- 검색 완전 실패 (0개 passage): {len(no_passages)}개")
print(f"- 검색 성공했지만 답변 실패: {len(has_passages)}개")

# Analyze Argentina case
print("\n" + "="*80)
print("ARGENTINA 케이스 상세 분석")
print("="*80)
argentina = [r for r in results if 'Argentina' in r['question']][0]

print(f"Question: {argentina['question']}")
print(f"Gold Answer: {argentina['gold_answer']}")
print(f"Predicted: {argentina['predicted_answer']}")
print(f"\nRetrieved Passages ({argentina['retrieved_passages']['count']}개):")
for title in argentina['retrieved_passages']['titles']:
    print(f"  - {title}")

print(f"\nSub-Question 분석:")
for sq in argentina['decomposition']['subquestions']:
    print(f"  {sq['id']}: {sq['question']}")
    print(f"    Answer: {sq['answer']}")
    
print(f"\nSub-Question별 Retrieval:")
for sq_ret in argentina['retrieved_passages']['by_subquestion']:
    print(f"  {sq_ret['subquestion_id']}: {sq_ret['titles']}")

# Check if Taquini Plan was retrieved
if 'Taquini Plan' in argentina['retrieved_passages']['titles']:
    print("\n✅ Taquini Plan이 검색되었습니다!")
else:
    print("\n❌ Taquini Plan이 검색되지 않았습니다!")

# Analyze some successful cases
print("\n" + "="*80)
print("성공한 케이스 샘플 (처음 3개)")
print("="*80)
success_cases = [r for r in results if r['exact_match']][:3]
for case in success_cases:
    print(f"\nQuestion: {case['question'][:80]}...")
    print(f"Gold Answer: {case['gold_answer']}")
    print(f"Predicted: {case['predicted_answer']}")
    print(f"Retrieved: {case['retrieved_passages']['count']}개 passages")
