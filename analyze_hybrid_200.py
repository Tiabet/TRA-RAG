"""
Analyze test_hybrid_200_results.json in detail
- Check which queries found supporting facts
- Find best/worst performing queries
- Analyze failure patterns
"""
import json


def analyze_results():
    print("="*80)
    print("DETAILED ANALYSIS - Hybrid Retrieval 200 Results")
    print("="*80)
    
    # Load results
    with open('test_hybrid_200_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_results = data['all_results']
    
    # Categorize queries by recall performance
    full_recall = []
    partial_recall = []
    no_recall = []
    no_retrieval = []
    
    for r in all_results:
        if not r['success']:
            no_retrieval.append(r)
            continue
        
        supporting_titles = [sf[0] for sf in r['supporting_facts']]
        retrieved_titles = [p['title'] for p in r['retrieved_passages']]
        
        matches = [st for st in supporting_titles if st in retrieved_titles]
        
        if len(matches) == len(supporting_titles) and len(supporting_titles) > 0:
            full_recall.append({**r, 'matches': matches, 'supporting_titles': supporting_titles})
        elif len(matches) > 0:
            partial_recall.append({**r, 'matches': matches, 'supporting_titles': supporting_titles})
        elif len(supporting_titles) > 0:
            no_recall.append({**r, 'matches': matches, 'supporting_titles': supporting_titles})
    
    print(f"\n📊 RECALL CATEGORIES:")
    print(f"  ✅ Full recall: {len(full_recall)} queries")
    print(f"  ⚠️  Partial recall: {len(partial_recall)} queries")
    print(f"  ❌ No recall: {len(no_recall)} queries")
    print(f"  🚫 No retrieval: {len(no_retrieval)} queries")
    
    # Show examples of full recall
    if full_recall:
        print("\n" + "="*80)
        print("✅ FULL RECALL EXAMPLES (Top 5)")
        print("="*80)
        for i, r in enumerate(full_recall[:5], 1):
            print(f"\n{i}. [{r['index']}] {r['question']}")
            print(f"   Type: {r['type']}")
            print(f"   Supporting facts: {r['supporting_titles']}")
            print(f"   Retrieved: {len(r['retrieved_passages'])} passages")
            print(f"   ✓ All {len(r['matches'])} supporting facts found!")
    
    # Show examples of partial recall
    if partial_recall:
        print("\n" + "="*80)
        print("⚠️  PARTIAL RECALL EXAMPLES (Top 5)")
        print("="*80)
        for i, r in enumerate(partial_recall[:5], 1):
            print(f"\n{i}. [{r['index']}] {r['question']}")
            print(f"   Type: {r['type']}")
            print(f"   Supporting facts: {r['supporting_titles']}")
            print(f"   Retrieved: {len(r['retrieved_passages'])} passages")
            print(f"   ✓ Found: {r['matches']} ({len(r['matches'])}/{len(r['supporting_titles'])})")
            missed = [st for st in r['supporting_titles'] if st not in r['matches']]
            print(f"   ✗ Missed: {missed}")
    
    # Show examples of no recall
    if no_recall:
        print("\n" + "="*80)
        print("❌ NO RECALL EXAMPLES (Top 5)")
        print("="*80)
        for i, r in enumerate(no_recall[:5], 1):
            print(f"\n{i}. [{r['index']}] {r['question']}")
            print(f"   Type: {r['type']}")
            print(f"   Supporting facts needed: {r['supporting_titles']}")
            print(f"   Retrieved: {len(r['retrieved_passages'])} passages")
            retrieved_sample = [p['title'] for p in r['retrieved_passages'][:3]]
            print(f"   Retrieved (sample): {retrieved_sample}")
            print(f"   ✗ No supporting facts found!")
    
    # Show no retrieval cases
    if no_retrieval:
        print("\n" + "="*80)
        print("🚫 NO RETRIEVAL EXAMPLES (Top 5)")
        print("="*80)
        for i, r in enumerate(no_retrieval[:5], 1):
            print(f"\n{i}. [{r['index']}] {r['question']}")
            print(f"   Type: {r['type']}")
            print(f"   Entities: {r['num_entities']}")
            print(f"   Error: {r.get('error', 'No passages retrieved')}")
    
    # Analyze by question type
    print("\n" + "="*80)
    print("📊 ANALYSIS BY QUESTION TYPE")
    print("="*80)
    
    for q_type in ['bridge', 'comparison']:
        type_results = [r for r in all_results if r['type'] == q_type]
        if not type_results:
            continue
        
        type_full = [r for r in full_recall if r['type'] == q_type]
        type_partial = [r for r in partial_recall if r['type'] == q_type]
        type_no = [r for r in no_recall if r['type'] == q_type]
        
        print(f"\n{q_type.upper()}:")
        print(f"  Total: {len(type_results)}")
        print(f"  Full recall: {len(type_full)} ({len(type_full)/len(type_results)*100:.1f}%)")
        print(f"  Partial recall: {len(type_partial)} ({len(type_partial)/len(type_results)*100:.1f}%)")
        print(f"  No recall: {len(type_no)} ({len(type_no)/len(type_results)*100:.1f}%)")
    
    # Find high-performing queries (many passages, high recall)
    print("\n" + "="*80)
    print("🌟 HIGH-PERFORMING QUERIES (Many passages + Full recall)")
    print("="*80)
    
    high_performing = sorted(
        [r for r in full_recall if r['num_passages'] >= 3],
        key=lambda x: x['num_passages'],
        reverse=True
    )[:5]
    
    for i, r in enumerate(high_performing, 1):
        print(f"\n{i}. [{r['index']}] {r['question'][:70]}...")
        print(f"   Retrieved: {r['num_passages']} passages (all {len(r['matches'])} supporting facts found)")
        print(f"   Stage 1-B candidates: {r['stage1b_candidates']} → LLM filtered: {r['stage1b_filtered']}")
    
    # Find queries with excessive passages
    print("\n" + "="*80)
    print("⚠️  EXCESSIVE RETRIEVAL (Too many passages)")
    print("="*80)
    
    excessive = sorted(
        [r for r in all_results if r['num_passages'] > 50],
        key=lambda x: x['num_passages'],
        reverse=True
    )[:5]
    
    for i, r in enumerate(excessive, 1):
        supporting_titles = [sf[0] for sf in r['supporting_facts']]
        retrieved_titles = [p['title'] for p in r['retrieved_passages']]
        matches = [st for st in supporting_titles if st in retrieved_titles]
        
        print(f"\n{i}. [{r['index']}] {r['question'][:70]}...")
        print(f"   Retrieved: {r['num_passages']} passages (too many!)")
        print(f"   Supporting facts found: {len(matches)}/{len(supporting_titles)}")
        print(f"   Stage 1-B: {r['stage1b_candidates']} candidates → {r['stage1b_filtered']} filtered")
        
        # Check if LLM filtering failed
        if r['stage1b_candidates'] > 0 and r['stage1b_filtered'] == r['stage1b_candidates']:
            print(f"   ⚠️  LLM filtering may have failed (fallback?)")


if __name__ == "__main__":
    try:
        analyze_results()
    except FileNotFoundError:
        print("Error: test_hybrid_200_results.json not found")
        print("Please run test_hybrid_200.py first")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
