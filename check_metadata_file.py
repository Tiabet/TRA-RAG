import json

# Check if metadata file exists and load it
try:
    with open('HotpotQA/hotpotqa_sample_200_pure_metadata_v2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ File found: HotpotQA/hotpotqa_sample_200_pure_metadata_v2.json")
    print(f"Total entries: {len(data)}")
    
    if data:
        print(f"\nFirst entry structure:")
        print(f"  Keys: {list(data[0].keys())}")
        print(f"  Title: {data[0]['title']}")
        print(f"  Metadata keys: {list(data[0]['metadata'].keys())[:15]}")
        
        print(f"\nSample metadata content:")
        sample_meta = data[0]['metadata']
        for key, value in list(sample_meta.items())[:5]:
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            elif isinstance(value, (list, dict)):
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"  {key}: {value}")
except FileNotFoundError:
    print("❌ File not found: HotpotQA/hotpotqa_sample_200_pure_metadata_v2.json")
    print("\nTrying alternative files...")
    
    import os
    hotpot_files = [f for f in os.listdir('HotpotQA/') if 'metadata' in f.lower()]
    print(f"Available metadata files in HotpotQA/:")
    for f in hotpot_files:
        print(f"  - {f}")
