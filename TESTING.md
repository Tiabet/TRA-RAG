# Testing Guide

## 메인 테스트 파일 (Main Tests)

### `test_hybrid_retrieval.py` ✨ NEW
**Hybrid retrieval 종합 테스트**
- 3가지 쿼리 테스트 (Stephen Graham, Argentina education, Bee Cliff)
- 다중 타입 추출 시스템 검증
- Entity extraction → Type matching → 정답 확인

```bash
python test_hybrid_retrieval.py
```

**출력 예시:**
```
Extracted entities:
  1. education system (attribute/important) - 3 types
     - Concept/EducationalSystem
     - Concept/SocialSystem
     - Concept/AcademicField

Retrieval summary:
  - education system: 23 candidates → 3 LLM → 4 final

Expected answers:
  [FOUND] Education in Argentina
  [FOUND] Free education
```

---

### `test_llm_filtered.py`
**LLM-filtered retrieval 테스트**
- 2-stage filtering (Type matching → LLM filtering)
- 20개 쿼리 배치 테스트

```bash
python test_llm_filtered.py
```

---

### `test_200_comprehensive.py`
**200개 질문 종합 테스트**
- HotpotQA 200 샘플 전체 평가
- 정확도, 재현율 계산

```bash
python test_200_comprehensive.py
```

---

### `test_db_retrieval.py`
**DB 검색 기능 테스트**
- FTS5 full-text search 테스트
- Type/subtype 기반 검색 테스트
- 메타데이터 구조 검증

```bash
python test_db_retrieval.py
```

---

## Archive 폴더 (tests_archive/)

개발 과정에서 생성된 디버깅/테스트 파일들:

### 디버깅 파일
- `debug_type_search.py` - 타입 매칭 0개 문제 진단
- `debug_argentina.py` - 정답 존재 여부 확인
- `debug_ghost_llm_filter.py` - LLM 필터링 디버깅

### 단위 테스트
- `simple_test_types.py` - Entity extraction만 테스트
- `quick_test_matching.py` - Type filtering만 테스트
- `minimal_test.py` - DB 검색만 테스트 (LLM 없음)
- `test_multiple_types.py` - 전체 파이프라인 테스트

### 출력 파일
- `test_output.txt` - 3가지 쿼리 테스트 결과
- `test_output_utf8.txt` - UTF-8 인코딩 버전

**자세한 내용**: `tests_archive/README.md` 참조

---

## 개발 히스토리

### Phase 1: 단일 타입 시스템
- 문제: LLM이 "AcademicField" 추출 → DB에 없음 → 매칭 실패
- 결과: Argentina education 쿼리에서 타입 매칭 0개

### Phase 2: 다중 타입 시스템 (현재)
- 해결: 엔티티당 2-3개 possible_types 추출
- 결과: 23개 candidates → 정답 발견 ✅

---

## Quick Start

```bash
# 1. Hybrid retrieval 테스트 (추천)
python test_hybrid_retrieval.py

# 2. 개별 컴포넌트 테스트
python test_db_retrieval.py        # DB 기능만
python test_llm_filtered.py        # LLM filtering만

# 3. 대규모 평가
python test_200_comprehensive.py   # 200개 질문
```

---

## 테스트 커버리지

| 컴포넌트 | 테스트 파일 | 상태 |
|---------|------------|------|
| Entity Extraction (다중 타입) | test_hybrid_retrieval.py | ✅ |
| Value Matching (FTS5) | test_db_retrieval.py | ✅ |
| Type Filtering (다중 타입) | test_hybrid_retrieval.py | ✅ |
| LLM Filtering | test_llm_filtered.py | ✅ |
| Hybrid Pipeline | test_hybrid_retrieval.py | ✅ |
| Large-scale Evaluation | test_200_comprehensive.py | ✅ |

---

## 문제 해결

### 테스트 실패 시
1. `.env` 파일 확인 (ALICE_OPENAI_KEY, ALICE_CHAT_URL)
2. `metadata_v2.db` 파일 존재 확인
3. Python 패키지 설치: `pip install openai python-dotenv tqdm`

### 디버깅
- archive 폴더의 단위 테스트 사용
- `simple_test_types.py` - Entity extraction만 확인
- `minimal_test.py` - DB 검색만 확인 (빠름)
