"""
Compare current results with previous backup
"""
import json

# Load previous results
with open('multihop_pipeline_200_results_backup.json', 'r', encoding='utf-8') as f:
    prev_data = json.load(f)

prev_summary = prev_data.get('summary', {})

print("=" * 80)
print("COMPARISON: PREVIOUS vs CURRENT RESULTS")
print("=" * 80)

print("\n📊 PREVIOUS RESULTS (Backup - 2025-10-22):")
print(f"  Exact Match (EM):    {prev_summary.get('exact_match_percentage', 0):.1f}%")
print(f"  Token F1:            {prev_summary.get('avg_token_f1', 0)*100:.1f}%")
print(f"  Retrieval Recall:    {prev_summary.get('retrieval_recall_percentage', 0):.1f}%")

# Load current results
with open('evaluation_results_summary.json', 'r', encoding='utf-8') as f:
    curr_summary = json.load(f)

curr_overall = curr_summary['overall']

print("\n📊 CURRENT RESULTS (After Transitive Dependency Fix):")
print(f"  Exact Match (EM):    {curr_overall['exact_match']:.1f}%")
print(f"  Token F1:            {curr_overall['token_f1']:.1f}%")
print(f"  Retrieval Recall:    {curr_overall['retrieval_recall']:.1f}%")

print("\n" + "=" * 80)
print("IMPROVEMENT ANALYSIS")
print("=" * 80)

prev_em = prev_summary.get('exact_match_percentage', 0)
prev_f1 = prev_summary.get('avg_token_f1', 0) * 100
prev_recall = prev_summary.get('retrieval_recall_percentage', 0)

curr_em = curr_overall['exact_match']
curr_f1 = curr_overall['token_f1']
curr_recall = curr_overall['retrieval_recall']

em_diff = curr_em - prev_em
f1_diff = curr_f1 - prev_f1
recall_diff = curr_recall - prev_recall

print(f"\n📈 Exact Match (EM):       {em_diff:+.1f}% ({prev_em:.1f}% → {curr_em:.1f}%)")
print(f"📈 Token F1:               {f1_diff:+.1f}% ({prev_f1:.1f}% → {curr_f1:.1f}%)")
print(f"📈 Retrieval Recall:       {recall_diff:+.1f}% ({prev_recall:.1f}% → {curr_recall:.1f}%)")

print("\n" + "=" * 80)
print("KEY OBSERVATIONS")
print("=" * 80)

if em_diff > 0:
    print(f"✅ EM improved by {em_diff:.1f}%!")
else:
    print(f"⚠️  EM decreased by {abs(em_diff):.1f}%")

if f1_diff > 0:
    print(f"✅ F1 improved by {f1_diff:.1f}%!")
else:
    print(f"⚠️  F1 decreased by {abs(f1_diff):.1f}%")

if recall_diff > 0:
    print(f"✅ Retrieval recall improved by {recall_diff:.1f}%!")
elif recall_diff < 0:
    print(f"⚠️  Retrieval recall decreased by {abs(recall_diff):.1f}%")
else:
    print(f"➡️  Retrieval recall stayed the same")

print(f"\n💡 Insufficient Info responses: {curr_overall['insufficient_info_count']} ({curr_overall['insufficient_info_count']/curr_overall['total_questions']*100:.1f}%)")
