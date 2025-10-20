# Test Archive

이 폴더는 다중 타입 추출 시스템 개발 과정에서 생성된 디버깅 및 테스트 파일들을 보관합니다.

## 개발 배경

**문제**: LLM이 추출한 단일 타입이 DB 스키마와 일치하지 않아 타입 매칭 실패 (예: LLM이 "AcademicField" 추출 → DB에는 "SocialSystem", "EducationalSystem"만 존재)

**해결**: 엔티티당 2-3개의 possible_types를 추출하여 여러 타입 조합을 시도

## 디버깅 파일 (debug_*.py)

### `debug_type_search.py`
- **목적**: 타입 매칭이 0개 반환되는 원인 파악
- **발견**: DB에 "Concept/AcademicField" 서브타입 없음
- **결과**: DB에는 "SocialSystem"(19개), "EducationalSystem"(4개)만 존재 확인

### `debug_argentina.py`
- **목적**: "Education in Argentina" 정답이 DB에 존재하는지 확인
- **발견**: Concept/SocialSystem으로 존재함
- **결과**: Value search에서는 찾았지만 타입 미스매치로 필터링 실패

### `debug_ghost_llm_filter.py`
- **목적**: LLM 필터링 로직 디버깅
- **사용**: 이전 개발 단계에서 사용

## 테스트 파일

### `simple_test_types.py`
- **목적**: Entity extraction만 단독 테스트
- **확인**: LLM이 possible_types 배열을 제대로 반환하는지 검증
- **결과**: ✅ "education system" → 3개 타입 성공적으로 추출

### `quick_test_matching.py`
- **목적**: Type filtering (Stage 1-B)만 단독 테스트
- **확인**: 여러 타입 조합이 제대로 시도되는지 검증
- **결과**: ✅ 23개 candidates 발견 (3가지 타입 시도)

### `minimal_test.py`
- **목적**: DB 타입 검색만 테스트 (LLM 없이 빠른 확인)
- **확인**: DB에서 각 타입별로 몇 개 매칭되는지 확인
- **결과**: EducationalSystem(4) + SocialSystem(19) + AcademicField(0) = 23개

### `test_multiple_types.py`
- **목적**: 전체 hybrid retrieval 파이프라인 종합 테스트
- **범위**: Entity extraction → Retrieval → 정답 확인
- **결과**: ✅ Argentina education 쿼리에서 정답 발견

## 테스트 출력 파일

### `test_output.txt` / `test_output_utf8.txt`
- 3가지 쿼리 종합 테스트 결과
- Argentina education, Stephen Graham, Bee Cliff 모두 성공 확인

## 개발 결과

### 이전 (단일 타입)
- Type matching: **0개**
- 정답 발견: **실패**

### 현재 (다중 타입)
- Type matching: **23개** (3가지 타입 시도)
- 정답 발견: **성공** (Education in Argentina #1위)
- Tried types: `EducationalSystem` → `SocialSystem` → `AcademicField`

## 주요 코드 변경

1. **Prompt/entity_extraction_prompt.py**
   - Output: `type/subtype` → `possible_types` 배열
   - 각 엔티티당 2-3개 타입 조합 제공

2. **hybrid_retrieval.py**
   - `stage1b_type_filtering()`: 모든 타입 조합 시도
   - `retrieve_for_entity_hybrid()`: possible_types 추출 및 전달

## 보관 이유

향후 참고를 위해 개발 과정과 검증 방법을 보존합니다.
실제 사용 테스트는 `test_hybrid_retrieval.py` 사용 권장.
