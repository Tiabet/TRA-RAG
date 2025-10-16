"""
MuSiQue 데이터 구조 분석
"""

import json

# 데이터 로드
musique = json.load(open('MuSiQue/MuSiQue_qa_sample_200_context.json', 'r', encoding='utf-8'))

print('='*80)
print('🔍 MuSiQue 데이터 구조 분석')
print('='*80)
print()

# 첫 번째 샘플
sample = musique[0]
print('【샘플 1】')
print('-'*80)
print(f'Question: {sample["question"]}')
print(f'Answer: {sample["answer"]}')
print(f'ID: {sample["id"]}')
print()

# Question decomposition 확인
if 'question_decomposition' in sample:
    qd = sample['question_decomposition']
    print('Question Decomposition:')
    print(f'  Type: {type(qd).__name__}')
    
    if isinstance(qd, list):
        print(f'  Length: {len(qd)}')
        for i, step in enumerate(qd, 1):
            print(f'  {i}. {step}')
    elif isinstance(qd, dict):
        print(json.dumps(qd, indent=4, ensure_ascii=False))
    else:
        print(f'  Value: {qd}')
    print()

# Supporting facts 확인
if 'supporting_facts' in sample:
    print('Supporting Facts:')
    sf = sample['supporting_facts']
    print(f'  Type: {type(sf).__name__}')
    if isinstance(sf, list) and len(sf) > 0:
        print(f'  Count: {len(sf)}')
        for i, fact in enumerate(sf[:3], 1):
            print(f'  {i}. {fact}')
        if len(sf) > 3:
            print(f'  ... and {len(sf) - 3} more')
    print()

print('='*80)
print()

# 더 많은 샘플 확인 (hop 수 분석)
print('【Multi-hop 통계】')
print('-'*80)

hop_counts = {}
decomposition_exists = 0
decomposition_types = {}

for item in musique:
    if 'question_decomposition' in item and item['question_decomposition']:
        decomposition_exists += 1
        qd = item['question_decomposition']
        
        qd_type = type(qd).__name__
        decomposition_types[qd_type] = decomposition_types.get(qd_type, 0) + 1
        
        if isinstance(qd, list):
            hop_count = len(qd)
            hop_counts[hop_count] = hop_counts.get(hop_count, 0) + 1

print(f'Question decomposition 존재: {decomposition_exists}/{len(musique)}개')
print()

print('Decomposition 타입:')
for dtype, count in sorted(decomposition_types.items()):
    print(f'  {dtype}: {count}개')
print()

if hop_counts:
    print('Hop 수 분포:')
    for hop, count in sorted(hop_counts.items()):
        pct = count / decomposition_exists * 100 if decomposition_exists > 0 else 0
        print(f'  {hop}-hop: {count}개 ({pct:.1f}%)')
print()

print('='*80)
print()

# 다양한 hop 예시
print('【Hop별 질문 예시】')
print('-'*80)

for hop in sorted(hop_counts.keys())[:4]:
    examples = [q for q in musique if isinstance(q.get('question_decomposition'), list) 
                and len(q['question_decomposition']) == hop]
    if examples:
        ex = examples[0]
        print(f'\n✦ {hop}-hop 질문:')
        print(f'  Q: {ex["question"]}')
        print(f'  A: {ex["answer"]}')
        print(f'  Decomposition:')
        for i, step in enumerate(ex['question_decomposition'], 1):
            # dict인 경우와 string인 경우 처리
            if isinstance(step, dict):
                q_text = step.get('question', step.get('id', str(step)))
                print(f'    {i}. {q_text}')
            else:
                print(f'    {i}. {step}')

print()
print('='*80)
