"""
질문 타입 분석 스크립트
"""

import json

# 데이터 로드
hotpot = json.load(open('HotpotQA/hotpotqa_sample_200.json', 'r', encoding='utf-8'))
wiki2 = json.load(open('2WikiMultihopQA/2WikiMultihopQA_sample_200.json', 'r', encoding='utf-8'))
musique = json.load(open('MuSiQue/MuSiQue_qa_sample_200_context.json', 'r', encoding='utf-8'))

print('='*80)
print('📊 데이터셋별 질문 타입 분석')
print('='*80)
print()

# HotpotQA
print('【1. HotpotQA】')
print('-'*40)
types_h = {}
for item in hotpot:
    qtype = item.get('type', 'N/A')
    types_h[qtype] = types_h.get(qtype, 0) + 1

for k, v in sorted(types_h.items()):
    print(f'  ✓ {k}: {v}개 ({v/len(hotpot)*100:.1f}%)')
print(f'  📌 총: {len(hotpot)}개')
print()

# 예시
bridge_h = [q for q in hotpot if q.get('type') == 'bridge'][0]
comp_h = [q for q in hotpot if q.get('type') == 'comparison'][0]

print('  📝 Bridge 예시:')
print(f'     Q: {bridge_h["question"]}')
print(f'     A: {bridge_h["answer"]}')
print()
print('  📝 Comparison 예시:')
print(f'     Q: {comp_h["question"]}')
print(f'     A: {comp_h["answer"]}')
print()

print('='*80)
print()

# 2WikiMultihopQA
print('【2. 2WikiMultihopQA】')
print('-'*40)
types_w = {}
for item in wiki2:
    qtype = item.get('type', 'N/A')
    types_w[qtype] = types_w.get(qtype, 0) + 1

for k, v in sorted(types_w.items()):
    print(f'  ✓ {k}: {v}개 ({v/len(wiki2)*100:.1f}%)')
print(f'  📌 총: {len(wiki2)}개')
print()

# 예시
for qtype in ['compositional', 'comparison', 'bridge_comparison', 'inference']:
    examples = [q for q in wiki2 if q.get('type') == qtype]
    if examples:
        example = examples[0]
        pct = len(examples) / len(wiki2) * 100
        print(f'  📝 {qtype.capitalize()} 예시:')
        print(f'     Q: {example["question"]}')
        print(f'     A: {example["answer"]}')
        print()

print('='*80)
print()

# MuSiQue
print('【3. MuSiQue】')
print('-'*40)
types_m = {}
for item in musique:
    qtype = item.get('type', '')
    if not qtype:
        qtype = '(empty)'
    types_m[qtype] = types_m.get(qtype, 0) + 1

for k, v in sorted(types_m.items()):
    print(f'  ✓ {k}: {v}개 ({v/len(musique)*100:.1f}%)')
print(f'  📌 총: {len(musique)}개')
print()

# MuSiQue 예시
print('  📝 질문 예시:')
print(f'     Q: {musique[0]["question"]}')
print(f'     A: {musique[0]["answer"]}')
print()

print('='*80)
print()

# 요약
print('📋 요약')
print('-'*40)
print(f'HotpotQA:        Bridge(79%), Comparison(21%)')
print(f'2WikiMultihopQA: Compositional(46%), Comparison(25%), Bridge_comparison(19%), Inference(11%)')
print(f'MuSiQue:         타입 정보 없음 (빈 필드)')
print()
print('='*80)
