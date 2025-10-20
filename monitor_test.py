"""
Monitor test_hybrid_200.py progress
"""
import json
import time
import os

def monitor():
    print("Monitoring test_hybrid_200.py progress...")
    print("="*60)
    
    last_check = None
    
    while True:
        if os.path.exists('test_hybrid_200_results.json'):
            try:
                with open('test_hybrid_200_results.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                stats = data.get('stats', {})
                elapsed = data.get('execution_time', 0)
                
                print(f"\n✅ TEST COMPLETED!")
                print(f"Execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
                print(f"\nResults:")
                print(f"  Total queries: {stats.get('total', 0)}")
                print(f"  Retrieved: {stats.get('retrieved', 0)} ({stats.get('retrieved', 0)/stats.get('total', 1)*100:.1f}%)")
                print(f"  No match: {stats.get('no_match', 0)}")
                print(f"\nStages:")
                print(f"  Stage 1-A (Value): {stats.get('stage1a_total', 0)}")
                print(f"  Stage 1-B (Type candidates): {stats.get('stage1b_candidates', 0)}")
                print(f"  Stage 1-B (LLM filtered): {stats.get('stage1b_filtered', 0)}")
                print(f"  Stage 2 (Final): {stats.get('stage2_final', 0)}")
                
                break
            except json.JSONDecodeError:
                print("File exists but incomplete, waiting...")
                time.sleep(5)
        else:
            current_time = time.strftime("%H:%M:%S")
            print(f"\r⏳ Running... {current_time}", end='', flush=True)
            time.sleep(3)

if __name__ == "__main__":
    monitor()
