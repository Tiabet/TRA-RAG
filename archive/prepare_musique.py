#!/usr/bin/env python3
"""
MuSiQue 데이터셋 준비 스크립트
==============================
1. 1000개 중 200개 샘플링
2. MuSiQue 형식을 HotpotQA 형식으로 변환 (build_metadata.py 호환)

MuSiQue 형식:
{
    "id": "2hop__13548_13529",
    "paragraphs": [
        {"idx": 0, "title": "...", "paragraph_text": "...", "is_supporting": true/false},
        ...
    ],
    "question": "...",
    "question_decomposition": [...],
    "answer": "...",
    "answerable": true
}

HotpotQA 형식 (출력):
{
    "_id": "2hop__13548_13529",
    "question": "...",
    "answer": "...",
    "context": [
        ["Title1", ["sentence1", "sentence2", ...]],
        ["Title2", ["sentence1", "sentence2", ...]],
        ...
    ],
    "supporting_facts": [["Title1", 0], ["Title2", 1], ...],  # is_supporting=true인 것들
    "question_decomposition": [...]  # MuSiQue 고유 필드 유지
}
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import re


def split_into_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분리
    간단한 규칙 기반 분리 (마침표, 물음표, 느낌표 기준)
    """
    # 문장 분리 패턴
    # 약어 등을 고려한 간단한 분리
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # 빈 문장 제거
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def convert_musique_to_hotpot_format(item: Dict) -> Dict:
    """
    MuSiQue 형식을 HotpotQA 형식으로 변환
    """
    # context 변환: paragraphs -> [[title, [sentences]], ...]
    context = []
    supporting_facts = []
    
    for para in item.get("paragraphs", []):
        title = para["title"]
        paragraph_text = para["paragraph_text"]
        is_supporting = para.get("is_supporting", False)
        
        # paragraph_text를 문장 리스트로 변환
        sentences = split_into_sentences(paragraph_text)
        
        context.append([title, sentences])
        
        # supporting_facts 생성 (is_supporting이 true인 경우)
        if is_supporting:
            # 모든 문장에 대해 supporting fact 추가
            for sent_idx in range(len(sentences)):
                supporting_facts.append([title, sent_idx])
    
    # 변환된 형식 반환
    converted = {
        "_id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "context": context,
        "supporting_facts": supporting_facts,
        # MuSiQue 고유 필드 유지
        "question_decomposition": item.get("question_decomposition", []),
        "answer_aliases": item.get("answer_aliases", []),
        "answerable": item.get("answerable", True)
    }
    
    return converted


def sample_and_convert(
    input_path: str,
    output_path: str,
    sample_size: int = 200,
    seed: int = 42,
    only_answerable: bool = True
):
    """
    MuSiQue 데이터셋에서 샘플링하고 형식 변환
    
    Args:
        input_path: 입력 musique.json 경로
        output_path: 출력 경로
        sample_size: 샘플 크기
        seed: 랜덤 시드
        only_answerable: answerable=true인 것만 샘플링
    """
    print(f"📂 Loading: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Total items: {len(data)}")
    
    # answerable 필터링
    if only_answerable:
        data = [item for item in data if item.get("answerable", True)]
        print(f"   Answerable items: {len(data)}")
    
    # 샘플링
    random.seed(seed)
    if len(data) > sample_size:
        sampled = random.sample(data, sample_size)
    else:
        sampled = data
        print(f"   ⚠️ Not enough data, using all {len(sampled)} items")
    
    print(f"   Sampled: {sample_size} items")
    
    # 형식 변환
    print(f"\n🔄 Converting to HotpotQA format...")
    converted = []
    
    for item in sampled:
        try:
            conv = convert_musique_to_hotpot_format(item)
            converted.append(conv)
        except Exception as e:
            print(f"   ❌ Error converting {item.get('id')}: {e}")
    
    print(f"   Converted: {len(converted)} items")
    
    # 저장
    print(f"\n💾 Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    
    # 통계 출력
    print(f"\n{'='*60}")
    print("📊 Dataset Statistics")
    print(f"{'='*60}")
    
    # hop 분포 분석
    hop_counts = {}
    for item in converted:
        item_id = item["_id"]
        # ID에서 hop 수 추출 (예: "2hop__xxx" -> 2)
        if "hop" in item_id:
            hop = item_id.split("hop")[0]
            hop_counts[hop] = hop_counts.get(hop, 0) + 1
    
    print(f"Hop distribution:")
    for hop, count in sorted(hop_counts.items()):
        print(f"  {hop}-hop: {count}")
    
    # context 통계
    total_passages = sum(len(item["context"]) for item in converted)
    avg_passages = total_passages / len(converted)
    
    total_supporting = sum(len(item["supporting_facts"]) for item in converted)
    
    print(f"\nPassage statistics:")
    print(f"  Total passages: {total_passages}")
    print(f"  Avg per question: {avg_passages:.1f}")
    print(f"  Total supporting facts: {total_supporting}")
    
    print(f"{'='*60}\n")
    
    return converted


def main():
    """메인 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare MuSiQue dataset")
    parser.add_argument("-i", "--input", default="MuSiQue/musique.json", help="Input file")
    parser.add_argument("-o", "--output", default="MuSiQue/musique_sample_200.json", help="Output file")
    parser.add_argument("-n", "--num-samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    sample_and_convert(
        input_path=args.input,
        output_path=args.output,
        sample_size=args.num_samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
