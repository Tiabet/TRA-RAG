import json
import os
from collections import Counter, defaultdict
import pandas as pd

def analyze_statistics():
    base_dir = r"c:\Development\ChunkRAG_v2"
    input_path = os.path.join(base_dir, "Analysis", "expanded_sd_links.json")
    
    print(f"Loading analysis file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Counters
    stats = {
        'rs': {
            'values': Counter(),
            'r_keys': Counter(),
            's_keys': Counter(),
            'key_pairs': Counter()
        },
        'rd': {
            'values': Counter(),
            'r_keys': Counter(),
            'd_keys': Counter(),
            'key_pairs': Counter()
        }
    }
    
    total_rs_links = 0
    total_rd_links = 0
    
    print("Aggregating statistics...")
    for item in data:
        # Process RS Links
        for entry in item.get('rs_shared_values', []):
            val = entry['value']
            stats['rs']['values'][val] += entry['pair_count'] # Weighted by how many pairs share this
            
            for link in entry.get('links', []):
                total_rs_links += 1
                r_keys = link.get('r_keys', [])
                s_keys = link.get('s_keys', [])
                
                for rk in r_keys:
                    stats['rs']['r_keys'][rk] += 1
                    for sk in s_keys:
                        stats['rs']['s_keys'][sk] += 1
                        stats['rs']['key_pairs'][f"{rk} -> {sk}"] += 1

        # Process RD Links
        for entry in item.get('rd_shared_values', []):
            val = entry['value']
            stats['rd']['values'][val] += entry['pair_count']
            
            for link in entry.get('links', []):
                total_rd_links += 1
                r_keys = link.get('r_keys', [])
                d_keys = link.get('d_keys', [])
                
                for rk in r_keys:
                    stats['rd']['r_keys'][rk] += 1
                    for dk in d_keys:
                        stats['rd']['d_keys'][dk] += 1
                        stats['rd']['key_pairs'][f"{rk} -> {dk}"] += 1

    print(f"\nTotal RS Links: {total_rs_links}")
    print(f"Total RD Links: {total_rd_links}")
    print(f"Ratio (RD/RS): {total_rd_links/total_rs_links:.2f}" if total_rs_links > 0 else "Ratio: N/A")

    # Helper to print top N
    def print_top_n(counter, label, n=20):
        print(f"\n--- Top {n} {label} ---")
        print(f"{'Item':<60} | {'Count':<8} | {'%':<6}")
        print("-" * 80)
        total = sum(counter.values())
        for item, count in counter.most_common(n):
            perc = (count / total) * 100 if total > 0 else 0
            print(f"{item[:60]:<60} | {count:<8} | {perc:.1f}%")

    # Helper to compare RS vs RD for a specific category (e.g., Values or Key Pairs)
    def compare_distributions(rs_counter, rd_counter, label, min_count=10):
        print(f"\n=== Comparative Analysis: {label} ===")
        print(f"{'Item':<60} | {'RS':<6} | {'RD':<6} | {'RS/(RS+RD)':<12}")
        print("-" * 95)
        
        # Get union of keys
        all_items = set(rs_counter.keys()) | set(rd_counter.keys())
        
        comparison = []
        for item in all_items:
            rs_c = rs_counter[item]
            rd_c = rd_counter[item]
            if rs_c + rd_c < min_count:
                continue
            
            ratio = rs_c / (rs_c + rd_c)
            comparison.append((item, rs_c, rd_c, ratio))
            
        # Sort by Ratio descending (Best for RS)
        comparison.sort(key=lambda x: x[3], reverse=True)
        
        print("\n[Top RS-leaning (High Precision)]")
        for item, rs, rd, ratio in comparison[:15]:
            print(f"{item[:60]:<60} | {rs:<6} | {rd:<6} | {ratio:.2f}")
            
        print("\n[Top RD-leaning (High Noise)]")
        # Sort by Ratio ascending (Worst for RS, mostly RD)
        comparison.sort(key=lambda x: x[3])
        for item, rs, rd, ratio in comparison[:15]:
            print(f"{item[:60]:<60} | {rs:<6} | {rd:<6} | {ratio:.2f}")

    # 1. Values Analysis
    print("\n\n################# VALUE ANALYSIS #################")
    print_top_n(stats['rs']['values'], "RS Values")
    print_top_n(stats['rd']['values'], "RD Values")
    compare_distributions(stats['rs']['values'], stats['rd']['values'], "Values", min_count=20)

    # 2. Key Pairs Analysis (The most important part for strategy)
    print("\n\n################# KEY PAIR ANALYSIS #################")
    print_top_n(stats['rs']['key_pairs'], "RS Key Pairs")
    print_top_n(stats['rd']['key_pairs'], "RD Key Pairs")
    compare_distributions(stats['rs']['key_pairs'], stats['rd']['key_pairs'], "Key Pairs", min_count=50)

if __name__ == "__main__":
    analyze_statistics()
