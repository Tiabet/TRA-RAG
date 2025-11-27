#!/usr/bin/env python3
"""Analyze why V3 (original passages) performs worse than V2 (metadata)."""

import json

# Load data
with open('Results/test_pipeline_v3_original_200_results.json', encoding='utf-8') as f:
    v3 = json.load(f)

with open('Results/test_new_pipeline_200_results_v2.json', encoding='utf-8') as f:
    v2 = json.load(f)

with open('HotpotQA/hotpotqa_sample_200.json', encoding='utf-8') as f:
    hotpot = json.load(f)

# Build title -> passage mapping from HotpotQA
original_passages = {}
for item in hotpot:
    for title, sents in item.get('context', []):
        if title not in original_passages:
            original_passages[title] = ''.join(sents)

print(f"Total unique titles in HotpotQA sample: {len(original_passages)}")

# Count "Insufficient information" answers
insuf_v2 = sum(1 for r in v2['results'] if r.get('predicted_answer') and 'insufficient' in r['predicted_answer'].lower())
insuf_v3 = sum(1 for r in v3['results'] if r.get('predicted_answer') and 'insufficient' in r['predicted_answer'].lower())

print(f"\nV2 (Metadata) 'Insufficient': {insuf_v2}")
print(f"V3 (Original) 'Insufficient': {insuf_v3}")

# Check specific cases where V2 was correct but V3 was wrong
print("\n" + "="*80)
print("Cases where V2 correct but V3 wrong:")
print("="*80)

count = 0
for i, (r2, r3) in enumerate(zip(v2['results'], v3['results'])):
    gold = r2['gold_answer'].lower()
    pred_v2 = (r2.get('predicted_answer') or '').lower()
    pred_v3 = (r3.get('predicted_answer') or '').lower()
    
    # Simple match check
    v2_correct = gold in pred_v2 or pred_v2 in gold
    v3_correct = gold in pred_v3 or pred_v3 in gold
    
    if v2_correct and not v3_correct and count < 5:
        count += 1
        print(f"\n[{i+1}] Question: {r2['question'][:80]}...")
        print(f"    Gold: {r2['gold_answer']}")
        print(f"    V2 pred: {pred_v2}")
        print(f"    V3 pred: {pred_v3}")

# Check if retrieved titles exist in original_passages
# We need to check the metadata DB titles vs original titles
import sqlite3

conn = sqlite3.connect('HotpotQA/metadata_v3.db')
cursor = conn.cursor()
cursor.execute("SELECT title FROM metadata")
db_titles = set(row[0] for row in cursor.fetchall())
conn.close()

original_titles = set(original_passages.keys())

print("\n" + "="*80)
print("Title Matching Analysis:")
print("="*80)
print(f"Titles in metadata_v3.db: {len(db_titles)}")
print(f"Titles in HotpotQA original: {len(original_titles)}")
print(f"Overlap: {len(db_titles & original_titles)}")
print(f"In DB but not in original: {len(db_titles - original_titles)}")
print(f"In original but not in DB: {len(original_titles - db_titles)}")

# Show some examples of mismatched titles
missing_from_original = db_titles - original_titles
if missing_from_original:
    print(f"\nSample titles in DB but not in original (first 10):")
    for t in list(missing_from_original)[:10]:
        print(f"  - {t}")
