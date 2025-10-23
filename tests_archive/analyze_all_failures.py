import json
import sqlite3

# Load checkpoint
with open('multihop_pipeline_200_checkpoint.json', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']

# Connect to DB
conn = sqlite3.connect('metadata_v2.db')
cursor = conn.cursor()

# Get insufficient cases
insufficient = [r for r in results if r.get('predicted_answer') == 'Insufficient information.']

print("="*80)
print(f"ANALYZING {len(insufficient)} INSUFFICIENT CASES")
print("="*80)

# Analyze first 5 cases in detail
for idx, case in enumerate(insufficient[:5]):
    print(f"\n{'='*80}")
    print(f"CASE {idx+1}: Q{case.get('processing_index')}")
    print(f"{'='*80}")
    print(f"Question: {case['question'][:100]}...")
    print(f"Gold Answer: {case['gold_answer']}")
    print()
    
    # Gold supporting facts
    gold_titles = set([sf[0] for sf in case['gold_supporting_facts']])
    retrieved_titles = set(case['retrieved_passages']['titles'])
    
    print(f"Gold Supporting Facts ({len(gold_titles)}):")
    for title in gold_titles:
        print(f"  - {title}")
    print()
    
    print(f"Retrieved Passages ({len(retrieved_titles)}):")
    for title in list(retrieved_titles)[:5]:
        print(f"  - {title}")
    if len(retrieved_titles) > 5:
        print(f"  ... and {len(retrieved_titles) - 5} more")
    print()
    
    # Check retrieval success
    missing = gold_titles - retrieved_titles
    found = gold_titles & retrieved_titles
    
    if found:
        print(f"✓ Found {len(found)}/{len(gold_titles)} gold passages:")
        for title in found:
            print(f"    {title}")
    else:
        print(f"✗ Found 0/{len(gold_titles)} gold passages")
    
    if missing:
        print(f"\n✗ Missing {len(missing)}/{len(gold_titles)} gold passages:")
        for title in missing:
            print(f"    {title}")
            # Check if it exists in DB
            cursor.execute("SELECT COUNT(*) FROM metadata WHERE title = ?", (title,))
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"      → EXISTS in DB")
            else:
                print(f"      → NOT in DB")
    
    # Analyze sub-questions
    print(f"\nSub-Question Analysis:")
    subqs = case.get('decomposition', {}).get('subquestions', [])
    for sq in subqs:
        status = "✗ FAILED" if sq['answer'] == 'Insufficient information.' else "✓ ANSWERED"
        print(f"  {sq['id']}: {status}")
        print(f"     Q: {sq['question'][:70]}...")
        print(f"     A: {sq['answer'][:70]}...")
        
        # Check passages for this SQ
        sq_passages = [p for p in case['retrieved_passages']['by_subquestion'] 
                      if p['subquestion_id'] == sq['id']]
        if sq_passages:
            sq_titles = sq_passages[0]['titles']
            print(f"     Passages: {len(sq_titles)}")
            # Check if gold is in this SQ's passages
            sq_has_gold = bool(gold_titles & set(sq_titles))
            if sq_has_gold:
                print(f"     → Has gold passage")
            elif sq['answer'] == 'Insufficient information.':
                print(f"     → No gold passage (explains failure)")

# Summary statistics
print(f"\n\n{'='*80}")
print("SUMMARY: Retrieval Analysis")
print(f"{'='*80}")

retrieval_stats = {
    'all_gold_found': 0,
    'some_gold_found': 0,
    'no_gold_found': 0,
    'gold_exists_in_db': 0,
    'gold_not_in_db': 0
}

for case in insufficient:
    gold_titles = set([sf[0] for sf in case['gold_supporting_facts']])
    retrieved_titles = set(case['retrieved_passages']['titles'])
    
    found = gold_titles & retrieved_titles
    missing = gold_titles - retrieved_titles
    
    if len(found) == len(gold_titles):
        retrieval_stats['all_gold_found'] += 1
    elif len(found) > 0:
        retrieval_stats['some_gold_found'] += 1
    else:
        retrieval_stats['no_gold_found'] += 1
    
    # Check DB for missing
    for title in missing:
        cursor.execute("SELECT COUNT(*) FROM metadata WHERE title = ?", (title,))
        if cursor.fetchone()[0] > 0:
            retrieval_stats['gold_exists_in_db'] += 1
        else:
            retrieval_stats['gold_not_in_db'] += 1

print(f"\nRetrieval Success:")
print(f"  All gold passages found: {retrieval_stats['all_gold_found']} ({retrieval_stats['all_gold_found']/len(insufficient)*100:.1f}%)")
print(f"  Some gold passages found: {retrieval_stats['some_gold_found']} ({retrieval_stats['some_gold_found']/len(insufficient)*100:.1f}%)")
print(f"  No gold passages found: {retrieval_stats['no_gold_found']} ({retrieval_stats['no_gold_found']/len(insufficient)*100:.1f}%)")

print(f"\nMissing Passages (total {retrieval_stats['gold_exists_in_db'] + retrieval_stats['gold_not_in_db']}):")
print(f"  Exists in DB: {retrieval_stats['gold_exists_in_db']} (retrieval failed)")
print(f"  Not in DB: {retrieval_stats['gold_not_in_db']} (data problem)")

# Failure pattern analysis
print(f"\n\n{'='*80}")
print("FAILURE PATTERN ANALYSIS")
print(f"{'='*80}")

patterns = {
    'first_sq_failed': 0,
    'middle_sq_failed': 0,
    'all_sq_failed': 0,
    'has_gold_but_failed': 0
}

for case in insufficient:
    subqs = case.get('decomposition', {}).get('subquestions', [])
    failed_sqs = [sq for sq in subqs if sq['answer'] == 'Insufficient information.']
    
    if len(failed_sqs) == len(subqs):
        patterns['all_sq_failed'] += 1
    elif failed_sqs and failed_sqs[0]['id'] == 'SQ1':
        patterns['first_sq_failed'] += 1
    elif failed_sqs:
        patterns['middle_sq_failed'] += 1
    
    # Check if has gold but still failed
    gold_titles = set([sf[0] for sf in case['gold_supporting_facts']])
    retrieved_titles = set(case['retrieved_passages']['titles'])
    if gold_titles & retrieved_titles:
        patterns['has_gold_but_failed'] += 1

print(f"\nFailure Patterns:")
print(f"  All SQs failed: {patterns['all_sq_failed']} ({patterns['all_sq_failed']/len(insufficient)*100:.1f}%)")
print(f"  First SQ failed (cascade): {patterns['first_sq_failed']} ({patterns['first_sq_failed']/len(insufficient)*100:.1f}%)")
print(f"  Middle SQ failed: {patterns['middle_sq_failed']} ({patterns['middle_sq_failed']/len(insufficient)*100:.1f}%)")
print(f"  Has gold passage but failed: {patterns['has_gold_but_failed']} ({patterns['has_gold_but_failed']/len(insufficient)*100:.1f}%)")

conn.close()

print(f"\n\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")
print("""
Key Findings:
1. If 'All gold passages found' is high but still failed → Answer Generation problem
2. If 'No gold passages found' is high → Retrieval problem (Entity Extraction)
3. If 'First SQ failed' is high → Cascade failure from bad decomposition
4. If 'Has gold but failed' is high → LLM can't extract answer from passage
""")
