import json
import os

def extract_qa_decomposition():
    input_path = 'Analysis/recoverable_musique_qa.json'
    output_path = 'Analysis/recoverable_qa_decomposition.json'

    print(f"Loading QA data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    extracted_data = []
    for item in qa_data:
        extracted_item = {
            'id': item.get('_id') or item.get('id'),
            'question': item.get('question'),
            'supporting_facts': item.get('supporting_facts', []),
            'question_decomposition': item.get('question_decomposition', [])
        }
        extracted_data.append(extracted_item)

    print(f"Extracted decomposition info for {len(extracted_data)} questions.")
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    extract_qa_decomposition()
