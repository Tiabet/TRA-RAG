import json

# Load checkpoint
with open('multihop_pipeline_200_checkpoint.json', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']
insufficient = [r for r in results if r.get('predicted_answer') == 'Insufficient information.']
correct = [r for r in results if r.get('predicted_answer') != 'Insufficient information.']

print(f"{'='*80}")
print(f"CHECKPOINT ANALYSIS: {len(results)} Questions")
print(f"{'='*80}")
print(f"✗ Insufficient Information: {len(insufficient)} ({len(insufficient)/len(results)*100:.1f}%)")
print(f"✓ Has Answer: {len(correct)} ({len(correct)/len(results)*100:.1f}%)")
print()

# Analyze passage count for insufficient cases
print(f"{'='*80}")
print("INSUFFICIENT CASES - Passage Analysis")
print(f"{'='*80}")

insufficient_with_passages = [r for r in insufficient if r.get('retrieved_passages', {}).get('count', 0) > 0]
insufficient_no_passages = [r for r in insufficient if r.get('retrieved_passages', {}).get('count', 0) == 0]

print(f"With passages retrieved: {len(insufficient_with_passages)}")
print(f"No passages retrieved: {len(insufficient_no_passages)}")
print()

# Show examples with passages but still insufficient
if insufficient_with_passages:
    print(f"Examples WITH passages but INSUFFICIENT answer:")
    print("-" * 80)
    for r in insufficient_with_passages[:3]:
        passages = r.get('retrieved_passages', {})
        print(f"\nQ{r.get('processing_index')}: {passages.get('count', 0)} passages")
        print(f"Question: {r['question'][:100]}...")
        print(f"Gold Answer: {r['gold_answer']}")
        print(f"Passage Titles: {', '.join(passages.get('titles', [])[:3])}")
        
        # Check sub-question answers
        subqs = r.get('decomposition', {}).get('subquestions', [])
        for sq in subqs:
            answer = sq.get('answer', '')
            if answer == 'Insufficient information.':
                print(f"  └─ {sq['id']}: '{sq['question'][:60]}...' → INSUFFICIENT")
        print()

# Analyze correct answers
print(f"\n{'='*80}")
print("CORRECT CASES - Passage Analysis")
print(f"{'='*80}")
correct_passage_counts = [r.get('retrieved_passages', {}).get('count', 0) for r in correct]
if correct_passage_counts:
    print(f"Avg passages: {sum(correct_passage_counts)/len(correct_passage_counts):.1f}")
    print(f"Min: {min(correct_passage_counts)}, Max: {max(correct_passage_counts)}")
    
    print(f"\nExamples WITH correct answers:")
    print("-" * 80)
    for r in correct[:2]:
        passages = r.get('retrieved_passages', {})
        print(f"\nQ{r.get('processing_index')}: {passages.get('count', 0)} passages")
        print(f"Question: {r['question'][:100]}...")
        print(f"Gold Answer: {r['gold_answer']}")
        print(f"Predicted: {r['predicted_answer']}")
        print(f"Passage Titles: {', '.join(passages.get('titles', [])[:3])}")

# Analyze by question type
print(f"\n{'='*80}")
print("BY QUESTION TYPE")
print(f"{'='*80}")
bridge_insuf = [r for r in insufficient if r.get('question_type') == 'bridge']
comp_insuf = [r for r in insufficient if r.get('question_type') == 'comparison']
bridge_total = [r for r in results if r.get('question_type') == 'bridge']
comp_total = [r for r in results if r.get('question_type') == 'comparison']

print(f"Bridge: {len(bridge_insuf)}/{len(bridge_total)} insufficient ({len(bridge_insuf)/len(bridge_total)*100:.1f}%)")
print(f"Comparison: {len(comp_insuf)}/{len(comp_total)} insufficient ({len(comp_insuf)/len(comp_total)*100:.1f}%)")
