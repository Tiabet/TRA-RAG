import json

with open('MuSiQue/musique_sample_200.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 첫 번째 예시 확인
item = data[0]
print('=== Example 1 ===')
print(f'ID: {item["_id"]}')
print(f'Question: {item["question"]}')
print(f'Answer: {item["answer"]}')
print(f'Context passages: {len(item["context"])}')
print(f'Supporting facts: {len(item["supporting_facts"])}')
print()
print('Supporting facts detail:')
for sf in item['supporting_facts'][:10]:
    print(f'  {sf}')

# 원본 MuSiQue에서 is_supporting 개수 확인
print('\n=== Original MuSiQue ===')
with open('MuSiQue/musique.json', 'r', encoding='utf-8') as f:
    orig = json.load(f)

orig_item = orig[0]
supporting_count = sum(1 for p in orig_item['paragraphs'] if p.get('is_supporting', False))
print(f'is_supporting=True count: {supporting_count}')
print(f'Total paragraphs: {len(orig_item["paragraphs"])}')

# 문제: 현재 로직은 모든 문장에 대해 supporting_facts를 추가함
# HotpotQA는 [title, sentence_idx] 형식으로, 특정 문장만 supporting
# 하지만 우리는 supporting passage의 모든 문장을 추가했음
