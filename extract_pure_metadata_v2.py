"""
Extract pure metadata (title + metadata only) from v2 metadata file
"""
import json

# Load the NEW metadata file
with open('HotpotQA/hotpotqa_sample_200_metadata_v2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract only title and metadata from context_metadata
pure_metadata = []
for item in data:
    for context_item in item.get('context_metadata', []):
        if 'error' in context_item:
            print(f"⚠️ Skipping failed metadata: {context_item['title']}")
            continue
            
        title = context_item['title']
        metadata = context_item['metadata']
        
        pure_metadata.append({
            'title': title,
            'metadata': metadata
        })

print(f"Total metadata entries: {len(pure_metadata)}")
print(f"Sample: {pure_metadata[0]['title']}")

# Save to new file
output_path = 'HotpotQA/hotpotqa_sample_200_pure_metadata_v2.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(pure_metadata, f, indent=2, ensure_ascii=False)

print(f"✓ Saved to {output_path}")
