import json

with open('multihop_pipeline_200_checkpoint.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results'][:10]
insufficient = [r for r in results if 'Insufficient' in r['predicted_answer']]
exact_match = [r for r in results if r['exact_match']]

print("="*80)
print(f"첫 10개 결과 (개선 버전)")
print("="*80)
print(f"- Exact Match: {len(exact_match)}/10 ({len(exact_match)/10*100:.1f}%)")
print(f"- Insufficient: {len(insufficient)}/10 ({len(insufficient)/10*100:.1f}%)")
print(f"- Token F1 > 0: {len([r for r in results if r['token_f1'] > 0])}/10")

print("\n" + "="*80)
print("개별 결과:")
print("="*80)

for i, r in enumerate(results, 1):
    status = "✅" if r['exact_match'] else "❌"
    print(f"\n{i}. {status} {r['question'][:70]}...")
    print(f"   예측: {r['predicted_answer']}")
    print(f"   정답: {r['gold_answer']}")
    if r['exact_match']:
        print(f"   🎉 정확!")

# Argentina 케이스 확인
argentina = [r for r in results if 'Argentina' in r['question']]
if argentina:
    print("\n" + "="*80)
    print("🔍 Argentina 케이스 분석:")
    print("="*80)
    r = argentina[0]
    print(f"Question: {r['question']}")
    print(f"Predicted: {r['predicted_answer']}")
    print(f"Gold: {r['gold_answer']}")
    print(f"Match: {r['exact_match']}")
    print(f"\nRetrieved Passages: {r['retrieved_passages']['count']}개")
    for title in r['retrieved_passages']['titles']:
        if 'Taquini' in title:
            print(f"  ⭐ {title}")
        else:
            print(f"  - {title}")
