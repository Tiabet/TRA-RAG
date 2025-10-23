import json

# Load checkpoint
with open('multihop_pipeline_200_checkpoint.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find Argentina case
argentina = [r for r in data['results'] if 'Argentina' in r['question']][0]

print("="*80)
print("ARGENTINA 케이스 - Entity Extraction 분석")
print("="*80)

print("\n📝 Question:")
print(argentina['question'])

print("\n🔍 Sub-Questions:")
for sq in argentina['decomposition']['subquestions']:
    print(f"\n{sq['id']}: {sq['question']}")
    print(f"  Answer: {sq['answer']}")

print("\n\n🎯 Extracted Entities:")
print(f"Total: {argentina['extracted_entities']['count']}개")
print(f"\nUnique names: {argentina['extracted_entities']['unique_names']}")

print("\n\n📦 Entities by Sub-Question:")
for entity in argentina['extracted_entities']['all']:
    sq_id = entity.get('subquestion_id', 'Unknown')
    name = entity.get('name', 'Unknown')
    role = entity.get('role', 'Unknown')
    types = entity.get('types', [])
    
    print(f"\n{sq_id} - {name} (role: {role})")
    for t in types:
        print(f"  • {t.get('type')}/{t.get('subtype')}")

print("\n\n🔎 SQ1 Entity 검색 시나리오:")
print("SQ1: What is the plan for free education in state institutions of Argentina...")
print("\nEntity 검색:")
print("  1. 'free education' → FTS 검색 → 'Free education' 1개 발견")
print("  2. 'Argentina' → FTS 검색 → 'Taquini Plan' (1위!), 'Education in Argentina' 등 13개 발견")
print("  3. 'state institutions' → FTS 검색 → ?")
print("  4. 'initial level', 'primary level', etc. → FTS 검색 → ?")

print("\n❓ 왜 'Taquini Plan'이 최종 결과에 없었을까?")
print("  가능성 1: 'Argentina' entity를 검색하지 않았다")
print("  가능성 2: 'Argentina' 검색 결과를 LLM이 필터링했다")
print("  가능성 3: Entity Extraction에서 'Argentina'를 추출하지 않았다")

print("\n✅ 실제 추출된 Entity 확인:")
sq1_entities = [e for e in argentina['extracted_entities']['all'] if e.get('subquestion_id') == 'SQ1']
print(f"SQ1에서 추출된 entity 이름들:")
for e in sq1_entities:
    print(f"  - {e.get('name')}")
