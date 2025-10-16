"""
Entity Extraction 테스트 스크립트
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from Prompt.entity_extraction_prompt import entity_extraction_prompt

# 환경 변수 로드
load_dotenv()

# LLM 클라이언트 초기화
client = OpenAI(
    base_url=os.getenv("ALICE_CHAT_URL"),
    api_key=os.getenv("ALICE_OPENAI_KEY")
)

# 테스트 쿼리들 (단일 엔티티 + 다중 엔티티)
test_queries = [
    # Single entity
    "What is the capital city of the country where the Eiffel Tower is located?",
    "In what year did the first iPhone get released by Apple Inc.?",
    
    # Multiple entities (comparison)
    "Which was released first, Windows 95 or Mac OS X?",
    "Who is older, Leonardo DiCaprio or Brad Pitt?",
    
    # Multiple entities (explicit subjects)
    "What is the distance between Tokyo and Seoul?",
    "Which university did both Barack Obama and Michelle Obama attend?"
]

print("="*80)
print("🧪 Entity Extraction 테스트")
print("="*80)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'─'*80}")
    print(f"Query {i}: {query}")
    print(f"{'─'*80}")
    
    # 프롬프트 생성
    prompt = entity_extraction_prompt.replace("{input}", query)
    
    # LLM 호출
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert entity extraction system. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=500
    )
    
    # 응답 파싱
    result_text = response.choices[0].message.content.strip()
    
    # JSON 파싱 시도
    try:
        # Markdown 코드 블록 제거
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(lines[1:-1])
        
        result = json.loads(result_text)
        
        print(f"✅ 추출 성공!")
        
        # entities 배열 처리
        if "entities" in result:
            entities = result["entities"]
            print(f"  엔티티 개수: {len(entities)}")
            for idx, entity in enumerate(entities, 1):
                print(f"  [{idx}] Entity Name: {entity.get('entity_name')}")
                print(f"      Type: {entity.get('type')}")
                print(f"      Subtype: {entity.get('subtype')}")
        # 이전 형식 호환성 (단일 entity_name)
        elif "entity_name" in result:
            print(f"  Entity Name: {result.get('entity_name')}")
            print(f"  Type: {result.get('type')}")
            print(f"  Subtype: {result.get('subtype')}")
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 실패: {str(e)}")
        print(f"Raw response:\n{result_text}")

print(f"\n{'='*80}")
print("테스트 완료!")
print(f"{'='*80}")
