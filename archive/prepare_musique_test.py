import json
import os

def main():
    input_path = 'MuSiQue/musique_sample_200_metadata.json'
    output_path = 'MuSiQue/musique_test_set.json'
    
    print(f"Reading from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_set = []
    for item in data:
        test_set.append({
            '_id': item['id'],
            'question': item['question'],
            'answer': item['answer'],
            'type': 'bridge', # Dummy type
            'level': 'hard'   # Dummy level
        })

    print(f"Writing {len(test_set)} items to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    print("Done.")

if __name__ == "__main__":
    main()
