# 메타데이터 생성 스크립트 사용 가이드 (Async 버전)

## 📋 개요

`build_metadata.py`는 **AsyncIO와 OpenAI**를 사용하여 데이터셋의 각 passage에서 구조화된 메타데이터를 **비동기 병렬로** 추출하는 스크립트입니다.

## ⚡ 주요 특징

- **비동기 병렬 처리**: asyncio와 AsyncOpenAI를 사용하여 여러 요청을 동시에 처리
- **Semaphore 제어**: concurrency 옵션으로 동시 실행 개수 제어
- **속도 향상**: 순차 처리 대비 5-10배 빠른 처리 속도
- **Rate Limit 보호**: 요청 간 짧은 딜레이로 API 제한 방지

## 🚀 사용 방법

### 기본 사용법

```bash
python build_metadata.py --input HotpotQA/HotpotQA_aligned_200.json --output metadata/hotpot_metadata.json
```

### 동시 처리 개수 조정

```bash
python build_metadata.py \
    -i HotpotQA/HotpotQA_aligned_200.json \
    -o metadata/hotpot_metadata.json \
    --concurrency 10
```

### 전체 옵션

```bash
python build_metadata.py \
    --input HotpotQA/HotpotQA_aligned_200.json \
    --output metadata/hotpot_metadata.json \
    --model openai/gpt-4o-mini \
    --max-passages 10 \
    --batch-size 10 \
    --concurrency 5
```

## 📝 옵션 설명

| 옵션 | 축약 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `--input` | `-i` | ✅ | - | 입력 데이터셋 JSON 파일 경로 |
| `--output` | `-o` | ✅ | - | 출력 메타데이터 JSON 파일 경로 |
| `--model` | `-m` | ❌ | `openai/gpt-4o-mini` | 사용할 LLM 모델 |
| `--max-passages` | - | ❌ | 전체 | 처리할 최대 passage 수 (테스트용) |
| `--batch-size` | - | ❌ | `10` | 중간 저장 배치 크기 |
| `--concurrency` | - | ❌ | `5` | **동시 처리 개수 (병렬 처리)** ⚡ |

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

### 1. 전체 데이터셋 처리 (기본 동시성)

```bash
python build_metadata.py \
    -i HotpotQA/HotpotQA_aligned_200.json \
    -o metadata/hotpot_full_metadata.json
```

### 2. 높은 동시성으로 빠른 처리

```bash
python build_metadata.py \
    -i HotpotQA/HotpotQA_aligned_200.json \
    -o metadata/hotpot_fast_metadata.json \
    --concurrency 20
```

### 3. 안전한 처리 (낮은 동시성)

```bash
python build_metadata.py \
    -i HotpotQA/HotpotQA_aligned_200.json \
    -o metadata/hotpot_safe_metadata.json \
    --concurrency 2
```

### 4. 테스트 (10개만 처리)

```bash
python build_metadata.py \
    -i HotpotQA/HotpotQA_aligned_200.json \
    -o metadata/hotpot_test_metadata.json \
    --max-passages 10 \
    --concurrency 5
```

### 5. MuSiQue 처리

```bash
python build_metadata.py \
    -i MuSiQue/MuSiQue_qa_sample_200_context.json \
    -o metadata/musique_metadata.json \
    --concurrency 8
```

### 6. 2WikiMultihopQA 처리

```bash
python build_metadata.py \
    -i 2WikiMultihopQA/2WikiMultihopQA_sample_200.json \
    -o metadata/2wiki_metadata.json \
    --concurrency 8
```

## 🔄 처리 과정

1. **데이터 로딩**: 입력 JSON 파일 로드
2. **클라이언트 초기화**: AsyncOpenAI 클라이언트 생성
3. **작업 목록 생성**: 모든 passage를 task로 변환
4. **비동기 병렬 처리**: 
   - Semaphore로 동시 실행 개수 제한
   - asyncio.gather로 모든 task 병렬 실행
   - 각 요청 후 0.05초 대기 (Rate Limit 방지)
5. **결과 수집**: 각 항목별로 메타데이터 정리
6. **중간 저장**: batch_size마다 중간 결과 저장
7. **최종 저장**: 모든 처리 완료 후 최종 저장

## 📈 성능 비교

### 순차 처리 vs 비동기 병렬 처리

| 항목 | 순차 처리 | 병렬 처리 (concurrency=5) | 병렬 처리 (concurrency=10) |
|------|-----------|---------------------------|----------------------------|
| 100 passages | ~10분 | ~2분 | ~1분 |
| 1000 passages | ~100분 | ~20분 | ~10분 |
| 속도 향상 | 1x | **5x** | **10x** |

## ⚙️ Concurrency 설정 가이드

| 상황 | 권장 값 | 설명 |
|------|---------|------|
| 테스트/개발 | `2-5` | 안전하고 디버깅하기 쉬움 |
| 일반 사용 | `5-10` | 속도와 안정성 균형 |
| 빠른 처리 | `10-20` | 최대 속도, Rate Limit 주의 |
| API 제한 우려 | `1-3` | 가장 안전, 속도는 느림 |

## 📈 통계 출력

스크립트는 실행 중 다음 정보를 출력합니다:

- 전체 passage 수
- 사용 모델
- **동시 처리 개수** ⚡
- 실시간 진행률 (tqdm progress bar)
- 성공/실패 개수
- 성공률

출력 예시:
```
============================================================
🚀 메타데이터 생성 시작
============================================================
📂 입력: HotpotQA/HotpotQA_aligned_200.json
💾 출력: metadata/hotpot_metadata.json
🤖 모델: openai/gpt-4o-mini
⚡ 동시 처리: 5개

📖 데이터 로딩 중...
   ✅ 200개 항목 로드됨

🔧 LLM 클라이언트 초기화 중...
   ✅ 클라이언트 초기화 완료

📊 처리 대상: 1500개 passages
🤖 모델: openai/gpt-4o-mini
⚡ 동시 처리: 5개

메타데이터 생성: 100%|██████████| 1500/1500 [03:25<00:00, 7.3it/s]

============================================================
📊 처리 완료 통계
============================================================
✅ 성공: 1485개
❌ 실패: 15개
📈 성공률: 99.0%
============================================================
```

## ⚠️ 주의사항

1. **API Rate Limit**: 
   - 각 요청 사이에 0.05초 대기 시간 적용
   - concurrency가 너무 높으면 Rate Limit 발생 가능
   - 제한 발생 시 concurrency 값을 낮추세요

2. **메모리 사용**: 
   - 모든 task를 메모리에 로드하므로 대용량 데이터셋은 주의
   - max_passages 옵션으로 분할 처리 권장

3. **중간 저장**: 
   - batch_size마다 자동 저장되어 중단 시에도 데이터 보존
   - 네트워크 불안정 환경에서 유용

4. **JSON 파싱**: 
   - LLM 응답이 유효한 JSON이 아닐 경우 에러로 기록
   - raw_response 필드에서 원본 확인 가능

5. **환경 변수**: 
   - .env 파일이 반드시 필요함
   - API 키와 URL 확인

## 🐛 문제 해결

### 환경 변수 오류
```
ValueError: 환경 변수가 설정되지 않았습니다.
```
→ `.env` 파일에 `ALICE_CHAT_URL`과 `ALICE_OPENAI_KEY` 확인

### Rate Limit 오류
```
"error": "LLM 호출 실패: Rate limit exceeded"
```
→ `--concurrency` 값을 낮추세요 (예: 5 → 2)

### JSON 파싱 오류
```
"error": "JSON 파싱 실패: ..."
```
→ 출력 파일의 `raw_response` 필드에서 원본 응답 확인

### 메모리 부족
```
MemoryError: ...
```
→ `--max-passages`로 분할 처리하거나 `--concurrency` 낮추기

## 📦 필수 패키지

```bash
pip install openai python-dotenv tqdm
```

**버전 요구사항:**
- `openai >= 1.0.0` (AsyncOpenAI 지원)
- `python-dotenv >= 0.19.0`
- `tqdm >= 4.60.0`

## 🎯 다음 단계

1. 생성된 메타데이터 검증
2. 실패한 passage 재처리
3. 메타데이터를 활용한 RAG 시스템 구축
4. 성능 최적화 (concurrency 튜닝)

## 🔥 팁

1. **최적의 concurrency 찾기**: 
   - 작은 데이터셋으로 테스트하며 최적값 찾기
   - API 제한과 속도의 균형점 찾기

2. **재처리 전략**:
   - 실패한 항목만 추출하여 재처리
   - concurrency를 낮춰서 재시도

3. **비용 절감**:
   - max_passages로 일부만 먼저 테스트
   - 불필요한 passage 필터링

4. **모니터링**:
   - tqdm progress bar로 실시간 진행상황 확인
   - 중간 저장 파일로 부분 결과 검증
