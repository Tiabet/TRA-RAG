import json

# Load checkpoint
with open('multihop_pipeline_200_checkpoint.json', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']

# Find Q1 (Argentina) case
q1 = [r for r in results if r.get('processing_index') == 1][0]

print("="*80)
print("ARGENTINA CASE - DETAILED ANALYSIS")
print("="*80)
print(f"Question: {q1['question']}")
print(f"Gold Answer: {q1['gold_answer']}")
print(f"Predicted: {q1['predicted_answer']}")
print()

print("Gold Supporting Facts:")
for sf in q1['gold_supporting_facts']:
    print(f"  - {sf[0]} (sentence {sf[1]})")
print()

print("Decomposition:")
for sq in q1['decomposition']['subquestions']:
    print(f"\n{sq['id']}: {sq['question']}")
    print(f"  Answer: {sq['answer']}")
    print(f"  Depends on: {sq.get('depends_on', [])}")
print()

print("Retrieved Passages by SubQuestion:")
for sq_data in q1['retrieved_passages']['by_subquestion']:
    print(f"\n{sq_data['subquestion_id']}: {len(sq_data['titles'])} passages")
    for title in sq_data['titles']:
        print(f"  - {title}")
print()

print("="*80)
print("CHECKING: Are gold supporting facts in retrieved passages?")
print("="*80)

gold_titles = set([sf[0] for sf in q1['gold_supporting_facts']])
retrieved_titles = set(q1['retrieved_passages']['titles'])

print(f"\nGold supporting fact titles: {gold_titles}")
print(f"Retrieved passage titles: {retrieved_titles}")
print()

for gold_title in gold_titles:
    if gold_title in retrieved_titles:
        print(f"✓ '{gold_title}' WAS RETRIEVED")
    else:
        print(f"✗ '{gold_title}' NOT RETRIEVED")
print()

# Check which subquestion retrieved which passage
print("="*80)
print("PASSAGE RETRIEVAL BY SUBQUESTION")
print("="*80)
for sq_data in q1['retrieved_passages']['by_subquestion']:
    sq_id = sq_data['subquestion_id']
    sq_titles = set(sq_data['titles'])
    
    # Find which gold fact this SQ should have found
    sq_obj = [s for s in q1['decomposition']['subquestions'] if s['id'] == sq_id][0]
    
    print(f"\n{sq_id}: {sq_obj['question']}")
    print(f"  Answer: {sq_obj['answer']}")
    print(f"  Retrieved {len(sq_titles)} passages:")
    for title in sq_data['titles']:
        is_gold = title in gold_titles
        marker = "★" if is_gold else " "
        print(f"    {marker} {title}")
    
    # Check if gold facts are missing
    missing_gold = gold_titles - sq_titles
    if missing_gold and sq_obj['answer'] == 'Insufficient information.':
        print(f"  ⚠️  Missing gold passages: {missing_gold}")
