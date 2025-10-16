"""
MuSiQue Context 형식 변환 스크립트
====================================
MuSiQue의 paragraphs 구조를 HotpotQA/2WikiMultihopQA와 동일한 context 형식으로 변환합니다.

변환 전:
"paragraphs": [
    {
        "idx": 0,
        "title": "Lake Helena",
        "paragraph_text": "Lake Helena is a body of water...",
        "is_supporting": false
    },
    ...
]

변환 후:
"context": [
    ["Lake Helena", ["Lake Helena is a body of water..."]],
    ...
]
"""

import json
from pathlib import Path
from collections import defaultdict


def convert_paragraphs_to_context(data):
    """
    MuSiQue의 paragraphs를 context 형식으로 변환
    
    같은 title을 가진 paragraph들을 그룹화하여
    [title, [sentence1, sentence2, ...]] 형식으로 변환
    """
    converted_data = []
    
    for item in data:
        # title별로 paragraph_text를 그룹화
        title_to_texts = defaultdict(list)
        
        if "paragraphs" in item:
            for para in item["paragraphs"]:
                title = para.get("title", "")
                text = para.get("paragraph_text", "")
                title_to_texts[title].append(text)
        
        # context 형식으로 변환: [[title, [texts]], ...]
        context = []
        for title, texts in title_to_texts.items():
            context.append([title, texts])
        
        # 새로운 항목 생성 (paragraphs를 context로 교체)
        new_item = {
            "id": item.get("id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "context": context,  # paragraphs 대신 context 사용
            "supporting_facts": item.get("supporting_facts", []),
            "type": item.get("type", ""),
            "level": item.get("level", ""),
        }
        
        # 추가 필드가 있으면 포함
        if "question_decomposition" in item:
            new_item["question_decomposition"] = item["question_decomposition"]
        if "answer_aliases" in item:
            new_item["answer_aliases"] = item["answer_aliases"]
        if "answerable" in item:
            new_item["answerable"] = item["answerable"]
        if "entity_ids" in item:
            new_item["entity_ids"] = item["entity_ids"]
        if "evidences" in item:
            new_item["evidences"] = item["evidences"]
        if "evidences_id" in item:
            new_item["evidences_id"] = item["evidences_id"]
        if "answer_id" in item:
            new_item["answer_id"] = item["answer_id"]
        
        converted_data.append(new_item)
    
    return converted_data


def main():
    """메인 실행 함수"""
    base_path = Path(__file__).parent
    
    # 입력 파일
    input_file = base_path / "MuSiQue" / "MuSiQue_qa_sample_200.json"
    
    # 출력 파일
    output_file = base_path / "MuSiQue" / "MuSiQue_qa_sample_200_context.json"
    
    print("🔄 MuSiQue 데이터셋 context 형식 변환 시작...\n")
    
    # 데이터 로드
    print(f"📂 입력 파일: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        musique_data = json.load(f)
    
    print(f"   📊 로드된 항목 수: {len(musique_data)}개\n")
    
    # 변환 실행
    print("⚙️  변환 중...")
    converted_data = convert_paragraphs_to_context(musique_data)
    
    # 결과 저장
    print(f"💾 저장 중: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ 변환 완료: {len(converted_data)}개 항목\n")
    
    # 통계 출력
    print("="*60)
    print("📊 변환 완료 통계")
    print("="*60)
    print(f"변환된 항목 수:     {len(converted_data)}개")
    
    # 샘플 출력
    if converted_data:
        print("\n" + "="*60)
        print("📄 샘플 데이터 (첫 번째 항목)")
        print("="*60)
        
        sample = converted_data[0]
        print(f"\nID: {sample['id']}")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"\nContext (총 {len(sample['context'])}개 항목):")
        
        # 처음 2개의 context 항목만 출력
        for i, ctx in enumerate(sample['context'][:2]):
            title = ctx[0]
            sentences = ctx[1]
            print(f"\n  [{i}] Title: {title}")
            print(f"      Sentences: {len(sentences)}개")
            if sentences:
                print(f"      First: {sentences[0][:100]}...")
        
        if len(sample['context']) > 2:
            print(f"\n  ... (나머지 {len(sample['context']) - 2}개 항목 생략)")
    
    print("\n" + "="*60)
    print("✨ 변환 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
