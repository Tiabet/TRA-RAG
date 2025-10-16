"""
1000개 원본 데이터 분석 스크립트
"""

import json

print('='*80)
print('📊 1000개 원본 데이터 분석')
print('='*80)
print()

# 1. HotpotQA (jsonl 형식)
print('【1. HotpotQA】')
print('-'*80)
hotpot_lines = []
with open('HotpotQA/hotpot.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        hotpot_lines.append(json.loads(line.strip()))

print(f'총 항목 수: {len(hotpot_lines)}개')
print(f'첫 번째 항목 필드: {list(hotpot_lines[0].keys())}')
print()

# 타입 분석
types_h = {}
for item in hotpot_lines:
    qtype = item.get('type', 'N/A')
    types_h[qtype] = types_h.get(qtype, 0) + 1

print('질문 타입 분포:')
for k, v in sorted(types_h.items()):
    print(f'  {k}: {v}개 ({v/len(hotpot_lines)*100:.1f}%)')
print()

# Context 통계
total_passages = sum(len(item.get('context', [])) for item in hotpot_lines)
print(f'총 passages: {total_passages}개')
print(f'평균 passages/항목: {total_passages/len(hotpot_lines):.1f}개')
print()

print('='*80)
print()

# 2. 2WikiMultihopQA
print('【2. 2WikiMultihopQA】')
print('-'*80)
wiki2 = json.load(open('2WikiMultihopQA/2wikimultihopqa.json', 'r', encoding='utf-8'))

print(f'총 항목 수: {len(wiki2)}개')
print(f'첫 번째 항목 필드: {list(wiki2[0].keys())}')
print()

# 타입 분석
types_w = {}
for item in wiki2:
    qtype = item.get('type', 'N/A')
    types_w[qtype] = types_w.get(qtype, 0) + 1

print('질문 타입 분포:')
for k, v in sorted(types_w.items()):
    print(f'  {k}: {v}개 ({v/len(wiki2)*100:.1f}%)')
print()

# Context 통계
total_passages = sum(len(item.get('context', [])) for item in wiki2)
print(f'총 passages: {total_passages}개')
print(f'평균 passages/항목: {total_passages/len(wiki2):.1f}개')
print()

print('='*80)
print()

# 3. MuSiQue
print('【3. MuSiQue】')
print('-'*80)
musique = json.load(open('MuSiQue/musique.json', 'r', encoding='utf-8'))

print(f'총 항목 수: {len(musique)}개')
print(f'첫 번째 항목 필드: {list(musique[0].keys())}')
print()

# Hop 분석 (question_decomposition 활용)
hop_counts = {}
decomposition_exists = 0

for item in musique:
    if 'question_decomposition' in item and item['question_decomposition']:
        decomposition_exists += 1
        qd = item['question_decomposition']
        
        if isinstance(qd, list):
            hop_count = len(qd)
            hop_counts[hop_count] = hop_counts.get(hop_count, 0) + 1

print(f'Question decomposition 존재: {decomposition_exists}/{len(musique)}개')
print()

if hop_counts:
    print('Hop 수 분포:')
    for hop, count in sorted(hop_counts.items()):
        pct = count / decomposition_exists * 100 if decomposition_exists > 0 else 0
        print(f'  {hop}-hop: {count}개 ({pct:.1f}%)')
print()

# Context 통계 (paragraphs 형식)
if 'paragraphs' in musique[0]:
    total_passages = sum(len(item.get('paragraphs', [])) for item in musique)
    print(f'총 paragraphs: {total_passages}개')
    print(f'평균 paragraphs/항목: {total_passages/len(musique):.1f}개')
elif 'context' in musique[0]:
    total_passages = sum(len(item.get('context', [])) for item in musique)
    print(f'총 contexts: {total_passages}개')
    print(f'평균 contexts/항목: {total_passages/len(musique):.1f}개')
print()

print('='*80)
print()

# 전체 통계 요약
print('📋 전체 요약')
print('-'*80)
print(f'HotpotQA:        {len(hotpot_lines):4d}개 | {total_passages:5d} passages')
print(f'2WikiMultihopQA: {len(wiki2):4d}개 | {sum(len(item.get("context", [])) for item in wiki2):5d} passages')

musique_passages = sum(len(item.get('paragraphs', item.get('context', []))) for item in musique)
print(f'MuSiQue:         {len(musique):4d}개 | {musique_passages:5d} passages')
print()

total_items = len(hotpot_lines) + len(wiki2) + len(musique)
total_all_passages = (sum(len(item.get('context', [])) for item in hotpot_lines) + 
                      sum(len(item.get('context', [])) for item in wiki2) + 
                      musique_passages)

print(f'총합:            {total_items:4d}개 | {total_all_passages:5d} passages')
print()
print('='*80)
