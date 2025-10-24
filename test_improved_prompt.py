"""
Test Improved Prompts on Perfect Recall but Wrong Answer Cases
================================================================
이전 평가에서 Perfect Recall (모든 supporting facts 찾음)이었지만
답변이 틀렸던 62개 케이스만 다시 테스트합니다.

배치 처리: 42 questions/batch, max_workers=100
"""

import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

from metadata_db import MetadataDB
from multihop_pipeline import process_multiple_questions

load_dotenv()


def load_perfect_recall_wrong_answer_cases():
    """
    Load cases that had Perfect Recall but wrong answers from previous analysis.
    """
    # Load the analysis file
    analysis_path = Path('perfect_recall_low_accuracy_analysis.json')
    
    if not analysis_path.exists():
        print(f"❌ Analysis file not found: {analysis_path}")
        print("Please run analyze_perfect_recall_low_accuracy.py first!")
        return []
    
    with analysis_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get the cases
    cases = data.get('cases', [])
    
    print(f"✅ Loaded {len(cases)} Perfect Recall but Wrong Answer cases")
    
    return cases


def load_original_question_data():
    """Load original question data with full details"""
    gold_path = Path('HotpotQA/hotpotqa_sample_200.json')
    
    if not gold_path.exists():
        print(f"❌ Gold file not found: {gold_path}")
        return {}
    
    with gold_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Build mapping: question_id -> full question data
    question_map = {}
    for item in data:
        qid = item.get('_id')
        question_map[qid] = item
    
    return question_map


def prepare_test_questions(cases, question_map):
    """
    Prepare test questions from cases.
    Only include cases that had perfect recall but wrong answers.
    """
    test_questions = []
    
    for case in cases:
        qid = case['qid']
        
        if qid not in question_map:
            print(f"⚠️  Question {qid} not found in original data")
            continue
        
        original = question_map[qid]
        
        # Build test question
        test_q = {
            '_id': qid,
            'question': case['question'],
            'answer': case['gold_answer'],
            'type': original.get('type', 'bridge'),
            'level': original.get('level', 'hard'),
            'supporting_facts': original.get('supporting_facts', []),
            'context': original.get('context', []),
            # Store for comparison
            'previous_prediction': case['predicted_answer'],
            'retrieved_titles': case['retrieved_titles']
        }
        
        test_questions.append(test_q)
    
    return test_questions


async def main():
    print("="*100)
    print("Testing Improved Prompts on Perfect Recall but Wrong Answer Cases")
    print("="*100)
    print("\n📋 Test Strategy:")
    print("  - Only test questions that had Perfect Recall but wrong answers (62 cases)")
    print("  - Skip questions that were already correct or had recall failures")
    print("  - Settings: batch_size=42, max_workers=100")
    print("  - Compare old vs new predictions")
    print("="*100)
    
    # Load cases
    print("\n[1/4] Loading Perfect Recall but Wrong Answer cases...")
    cases = load_perfect_recall_wrong_answer_cases()
    
    if not cases:
        print("❌ No cases found. Exiting.")
        return
    
    print(f"✅ Found {len(cases)} cases to retest")
    
    # Load original question data
    print("\n[2/4] Loading original question data...")
    question_map = load_original_question_data()
    print(f"✅ Loaded {len(question_map)} original questions")
    
    # Prepare test questions
    print("\n[3/4] Preparing test questions...")
    test_questions = prepare_test_questions(cases, question_map)
    print(f"✅ Prepared {len(test_questions)} test questions")
    
    # Initialize
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    db = MetadataDB('metadata_v2.db')
    
    # Run evaluation with improved prompts
    print("\n[4/4] Running evaluation with improved prompts...")
    print(f"{'='*100}")
    print(f"Settings: batch_size=42, max_workers=100")
    print(f"{'='*100}\n")
    
    results = await process_multiple_questions(
        client, db, test_questions,
        max_workers=100,
        batch_size=42,
        use_fts=True,
        apply_llm_filter_stage1a=True,
        verbose=False
    )
    
    db.close()
    
    # Debug: Check first result structure
    if results:
        print("\n[DEBUG] First result keys:", list(results[0].keys()))
        print("[DEBUG] First result preview:")
        for key in ['predicted_answer', 'final_answer', 'answer']:
            if key in results[0]:
                print(f"  {key}: {results[0][key][:100] if isinstance(results[0][key], str) else results[0][key]}")
    
    # Analyze improvements
    print("\n" + "="*100)
    print("IMPROVEMENT ANALYSIS")
    print("="*100)
    
    improvements = 0
    still_wrong = 0
    new_errors = 0
    
    comparison_results = []
    
    for i, (test_q, result) in enumerate(zip(test_questions, results)):
        gold = test_q['answer'].lower().strip()
        old_pred = test_q['previous_prediction'].lower().strip()
        
        # Extract predicted answer - check different possible keys
        new_pred = result.get('predicted_answer') or result.get('final_answer') or result.get('answer') or ''
        new_pred = new_pred.lower().strip()
        
        # Normalize for comparison - USE SAME LOGIC AS ORIGINAL ANALYSIS
        # Original: gold in pred (not bidirectional!)
        old_correct = False
        new_correct = False
        
        if gold and old_pred and old_pred != 'n/a':
            old_correct = gold in old_pred  # Only check if gold is IN prediction
        
        if gold and new_pred and new_pred != 'n/a':
            new_correct = gold in new_pred  # Only check if gold is IN prediction
        
        if not old_correct and new_correct:
            improvements += 1
            status = "✅ IMPROVED"
        elif not old_correct and not new_correct:
            still_wrong += 1
            status = "❌ STILL WRONG"
        elif old_correct and not new_correct:
            new_errors += 1
            status = "⚠️  NEW ERROR"
        else:
            status = "✓ Already correct"
        
        # Store actual prediction (not lowercased)
        actual_new_pred = result.get('predicted_answer') or result.get('final_answer') or result.get('answer') or 'N/A'
        
        comparison_results.append({
            'qid': test_q['_id'],
            'question': test_q['question'],
            'gold_answer': test_q['answer'],
            'old_prediction': test_q['previous_prediction'],
            'new_prediction': actual_new_pred,
            'status': status,
            'old_correct': old_correct,
            'new_correct': new_correct
        })
    
    # Summary
    total = len(test_questions)
    print(f"\nTotal Cases Retested: {total}")
    print(f"  ✅ Improved (Wrong → Correct): {improvements} ({improvements/total*100:.1f}%)")
    print(f"  ❌ Still Wrong: {still_wrong} ({still_wrong/total*100:.1f}%)")
    print(f"  ⚠️  New Errors (shouldn't happen): {new_errors} ({new_errors/total*100:.1f}%)")
    
    improvement_rate = improvements / total * 100 if total > 0 else 0
    print(f"\n🎯 Improvement Rate: {improvement_rate:.1f}%")
    
    # Show some examples
    print(f"\n{'='*100}")
    print("EXAMPLES OF IMPROVEMENTS")
    print(f"{'='*100}")
    
    improved_cases = [r for r in comparison_results if r['status'] == "✅ IMPROVED"]
    
    for i, case in enumerate(improved_cases[:5], 1):
        print(f"\n{'-'*100}")
        print(f"Example {i}/{len(improved_cases)}")
        print(f"{'-'*100}")
        print(f"Question: {case['question']}")
        print(f"Gold Answer: {case['gold_answer']}")
        print(f"Old Prediction: {case['old_prediction']} ❌")
        print(f"New Prediction: {case['new_prediction']} ✅")
    
    # Show some still wrong cases
    if still_wrong > 0:
        print(f"\n{'='*100}")
        print("EXAMPLES OF STILL WRONG CASES (Need Further Investigation)")
        print(f"{'='*100}")
        
        wrong_cases = [r for r in comparison_results if r['status'] == "❌ STILL WRONG"]
        
        for i, case in enumerate(wrong_cases[:5], 1):
            print(f"\n{'-'*100}")
            print(f"Example {i}/{len(wrong_cases)}")
            print(f"{'-'*100}")
            print(f"Question: {case['question']}")
            print(f"Gold Answer: {case['gold_answer']}")
            print(f"Old Prediction: {case['old_prediction']} ❌")
            print(f"New Prediction: {case['new_prediction']} ❌")
    
    # Save results
    output_path = Path('improved_prompt_test_results.json')
    with output_path.open('w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_cases': total,
                'improvements': improvements,
                'still_wrong': still_wrong,
                'new_errors': new_errors,
                'improvement_rate': improvement_rate
            },
            'results': comparison_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*100}")
    print(f"✅ Detailed results saved to: {output_path}")
    print(f"{'='*100}")


if __name__ == "__main__":
    asyncio.run(main())
