import json
from collections import Counter

# Load data
with open('HotpotQA/hotpotqa_sample_200_entities.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total questions: {len(data)}')
success_count = sum(1 for d in data if d['extraction_result']['success'])
print(f'Success rate: {success_count}/{len(data)} ({success_count/len(data)*100:.1f}%)')

print('\nEntity count distribution:')
entity_counts = Counter(len(d['extraction_result']['entities']) for d in data if d['extraction_result']['success'])
print(f'  1 entity: {entity_counts[1]} ({entity_counts[1]/len(data)*100:.1f}%)')
print(f'  2 entities: {entity_counts[2]} ({entity_counts[2]/len(data)*100:.1f}%)')
three_plus = sum(v for k, v in entity_counts.items() if k >= 3)
print(f'  3+ entities: {three_plus} ({three_plus/len(data)*100:.1f}%)')

print('\nQuestion type vs Entity count:')
bridge = [d for d in data if d['type'] == 'bridge']
comp = [d for d in data if d['type'] == 'comparison']

bridge_entity_count = sum(len(d['extraction_result']['entities']) for d in bridge)
comp_entity_count = sum(len(d['extraction_result']['entities']) for d in comp)

print(f'  Bridge (n={len(bridge)}): avg {bridge_entity_count/len(bridge):.2f} entities/question')
print(f'  Comparison (n={len(comp)}): avg {comp_entity_count/len(comp):.2f} entities/question')

# Find some comparison examples
print('\nSample comparison questions with multiple entities:')
comp_multi = [d for d in comp if len(d['extraction_result']['entities']) > 1][:5]
for i, item in enumerate(comp_multi, 1):
    print(f'\n{i}. {item["question"][:100]}...')
    for ent in item['extraction_result']['entities']:
        print(f'   - {ent["entity_name"]} ({ent["type"]}/{ent.get("subtype", "N/A")})')

# Entity type distribution
print('\n\nEntity type distribution:')
all_entities = []
for d in data:
    if d['extraction_result']['success']:
        all_entities.extend(d['extraction_result']['entities'])

type_counts = Counter(e['type'] for e in all_entities)
for etype, count in type_counts.most_common():
    print(f'  {etype}: {count} ({count/len(all_entities)*100:.1f}%)')
