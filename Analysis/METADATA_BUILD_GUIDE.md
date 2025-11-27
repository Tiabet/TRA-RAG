# 메타데이터 생성 스크립트 사용 가이드

## 📋 개요

`build_metadata.py`는 LLM을 사용하여 데이터셋의 각 passage에서 구조화된 메타데이터를 추출하는 스크립트입니다.

## 🚀 사용 방법

### 기본 사용법

```bash
python build_metadata.py --input HotpotQA/HotpotQA_aligned_200.json --output metadata/hotpot_metadata.json
```

### 축약 옵션

```bash
python build_metadata.py -i HotpotQA/HotpotQA_aligned_200.json -o metadata/hotpot_metadata.json
```

### 전체 옵션

```bash
python build_metadata.py \
    --input HotpotQA/HotpotQA_aligned_200.json \
    --output metadata/hotpot_metadata.json \
    --model openai/gpt-4o-mini \
    --max-passages 10 \
    --batch-size 10
```

## 📝 옵션 설명

| 옵션 | 축약 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `--input` | `-i` | ✅ | - | 입력 데이터셋 JSON 파일 경로 |
| `--output` | `-o` | ✅ | - | 출력 메타데이터 JSON 파일 경로 |
| `--model` | `-m` | ❌ | `openai/gpt-4o-mini` | 사용할 LLM 모델 |
| `--max-passages` | - | ❌ | 전체 | 처리할 최대 passage 수 (테스트용) |
| `--batch-size` | - | ❌ | `10` | 중간 저장 배치 크기 |

## 🔧 환경 변수 설정

`.env` 파일에 다음 변수들이 설정되어 있어야 합니다:

```properties
ALICE_CHAT_URL=https://mlapi.run/40cc17ae-a89b-4f12-a7d6-13293180fc87/v1
ALICE_OPENAI_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## 📊 입력 형식

입력 데이터셋은 다음과 같은 구조여야 합니다:

```json
[
  {
    "_id": "5a77c1d95542995d831812a6",
    "question": "...",
    "answer": "...",
    "context": [
      [
        "Title 1",
        ["Sentence 1", "Sentence 2", ...]
      ],
      [
        "Title 2",
        ["Sentence 1", "Sentence 2", ...]
      ]
    ]
  }
]
```

## 📤 출력 형식

출력 파일은 다음과 같은 구조를 갖습니다:

```json
[
  {
    "id": "5a77c1d95542995d831812a6",
    "question": "...",
    "answer": "...",
    "context_metadata": [
      {
        "title": "Title 1",
        "metadata": {
          "title": "Title 1",
          "type": "Location",
          "subtype": "City",
          "attributes": { ... },
          "relations": [ ... ]
        }
      },
      {
        "title": "Title 2",
        "error": "JSON 파싱 실패: ...",
        "raw_response": "..."
      }
    ]
  }
]
```

## 💡 예제

### 1. 전체 데이터셋 처리

```bash
python build_metadata.py \
    -i HotpotQA/HotpotQA_aligned_200.json \
    -o metadata/hotpot_full_metadata.json
```

### 2. 테스트 (10개만 처리)

```bash
python build_metadata.py \
    -i HotpotQA/HotpotQA_aligned_200.json \
    -o metadata/hotpot_test_metadata.json \
    --max-passages 10
```

### 3. MuSiQue 처리

```bash
python build_metadata.py \
    -i MuSiQue/MuSiQue_qa_sample_200_context.json \
    -o metadata/musique_metadata.json
```

### 4. 2WikiMultihopQA 처리

```bash
python build_metadata.py \
    -i 2WikiMultihopQA/2WikiMultihopQA_sample_200.json \
    -o metadata/2wiki_metadata.json
```

## 🔄 처리 과정

1. **데이터 로딩**: 입력 JSON 파일 로드
2. **클라이언트 초기화**: .env에서 API 정보 로드
3. **Passage 처리**: 각 passage를 순회하며 메타데이터 생성
   - `[title, [sentences]]` 형식을 프롬프트에 삽입
   - LLM 호출 및 응답 파싱
   - 성공/실패 기록
4. **중간 저장**: batch_size마다 중간 결과 저장
5. **최종 저장**: 모든 처리 완료 후 최종 저장

## 📈 통계 출력

스크립트는 실행 중 다음 정보를 출력합니다:

- 전체 passage 수
- 실시간 진행률 (tqdm progress bar)
- 성공/실패 개수
- 성공률

## ⚠️ 주의사항

1. **API Rate Limit**: 각 요청 사이에 0.1초 대기 시간 적용
2. **중간 저장**: batch_size마다 자동 저장되어 중단 시에도 데이터 보존
3. **JSON 파싱**: LLM 응답이 유효한 JSON이 아닐 경우 에러로 기록
4. **환경 변수**: .env 파일이 반드시 필요함

## 🐛 문제 해결

### 환경 변수 오류
```
ValueError: 환경 변수가 설정되지 않았습니다.
```
→ `.env` 파일에 `ALICE_CHAT_URL`과 `ALICE_OPENAI_KEY` 확인

### JSON 파싱 오류
```
"error": "JSON 파싱 실패: ..."
```
→ 출력 파일의 `raw_response` 필드에서 원본 응답 확인

### API 호출 실패
```
"error": "LLM 호출 실패: ..."
```
→ API 키와 URL이 올바른지, 네트워크 연결 확인

## 📦 필수 패키지

```bash
pip install openai python-dotenv tqdm
```

또는

```bash
pip install -r requirements.txt
```

## 🎯 다음 단계

1. 생성된 메타데이터 검증
2. 실패한 passage 재처리
3. 메타데이터를 활용한 RAG 시스템 구축
