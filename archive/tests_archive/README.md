# Tests Archive# Test Archive



이 폴더는 개발 과정에서 작성한 테스트, 디버깅, 분석 스크립트를 보관합니다.이 폴더는 다중 타입 추출 시스템 개발 과정에서 생성된 디버깅 및 테스트 파일들을 보관합니다.



## 📁 파일 분류## 개발 배경



### 🔍 Argentina Case 디버깅 (Type Filter Bug 발견)**문제**: LLM이 추출한 단일 타입이 DB 스키마와 일치하지 않아 타입 매칭 실패 (예: LLM이 "AcademicField" 추출 → DB에는 "SocialSystem", "EducationalSystem"만 존재)

- `debug_argentina.py` - 초기 Argentina 케이스 디버깅

- `analyze_argentina_case.py` - Argentina 케이스 상세 분석**해결**: 엔티티당 2-3개의 possible_types를 추출하여 여러 타입 조합을 시도

- `analyze_argentina_detail.py` - SQ별 retrieved passages 분석

- `analyze_argentina_entities.py` - Entity extraction 분석## 디버깅 파일 (debug_*.py)

- `reproduce_argentina_full.py` - 전체 파이프라인 재현 스크립트

- `trace_argentina_full.py` - Argentina 케이스 트레이싱### `debug_type_search.py`

- **목적**: 타입 매칭이 0개 반환되는 원인 파악

### 📊 평가 결과 분석- **발견**: DB에 "Concept/AcademicField" 서브타입 없음

- `analyze_120_results.py` - 초기 120개 결과 분석 (78.3% failure)- **결과**: DB에는 "SocialSystem"(19개), "EducationalSystem"(4개)만 존재 확인

- `analyze_all_failures.py` - 전체 실패 케이스 분석

- `analyze_insufficient.py` - "Insufficient information" 케이스 분석### `debug_argentina.py`

- `quick_check_first10.py` - 첫 10개 결과 빠른 체크- **목적**: "Education in Argentina" 정답이 DB에 존재하는지 확인

- `analyze_hybrid_200.py` - Hybrid retrieval 200개 평가 분석- **발견**: Concept/SocialSystem으로 존재함

- **결과**: Value search에서는 찾았지만 타입 미스매치로 필터링 실패

### 🧪 Hybrid Retrieval 테스트

- `test_hybrid_200.py` - 200개 질문 hybrid retrieval 테스트### `debug_ghost_llm_filter.py`

- `test_hybrid_200_results.json` - 테스트 결과 데이터- **목적**: LLM 필터링 로직 디버깅

- `test_hybrid_200_summary.txt` - 테스트 요약- **사용**: 이전 개발 단계에서 사용



### 🔧 컴포넌트 단위 테스트## 테스트 파일

- `test_context_building.py` - build_context_from_previous() 테스트

- `test_decomposition.py` - Query decomposition 테스트### `simple_test_types.py`

- `test_decomposition_results.json` - Decomposition 결과- **목적**: Entity extraction만 단독 테스트

- `test_multiple_types.py` - 다중 타입 검색 테스트- **확인**: LLM이 possible_types 배열을 제대로 반환하는지 검증

- `simple_test_types.py` - 타입 검색 단순 테스트- **결과**: ✅ "education system" → 3개 타입 성공적으로 추출

- `minimal_test.py` - 최소 기능 테스트

- `quick_test_matching.py` - 빠른 매칭 테스트### `quick_test_matching.py`

- **목적**: Type filtering (Stage 1-B)만 단독 테스트

### 🐛 초기 디버깅 스크립트- **확인**: 여러 타입 조합이 제대로 시도되는지 검증

- `debug_ghost_llm_filter.py` - LLM 필터링 버그 디버깅- **결과**: ✅ 23개 candidates 발견 (3가지 타입 시도)

- `debug_type_search.py` - Type 검색 디버깅

### `minimal_test.py`

### 📄 테스트 출력- **목적**: DB 타입 검색만 테스트 (LLM 없이 빠른 확인)

- `test_output.txt` - 테스트 실행 출력- **확인**: DB에서 각 타입별로 몇 개 매칭되는지 확인

- `test_output_utf8.txt` - UTF-8 인코딩 테스트 출력- **결과**: EducationalSystem(4) + SocialSystem(19) + AcademicField(0) = 23개



## 🎯 주요 발견 사항### `test_multiple_types.py`

- **목적**: 전체 hybrid retrieval 파이프라인 종합 테스트

### Critical Bug #1: Type Filter in Stage 1-A- **범위**: Entity extraction → Retrieval → 정답 확인

- **발견**: `reproduce_argentina_full.py`로 Argentina 케이스 재현 시 발견- **결과**: ✅ Argentina education 쿼리에서 정답 발견

- **문제**: Stage 1-A (value-based search)에서 entity_type 필터를 적용해 cross-type 매칭 실패

- **예시**: "Argentina" (Location/Country) 검색 시 "Taquini Plan" (Concept/EducationalReform) 제외됨## 테스트 출력 파일

- **해결**: `hybrid_retrieval.py`에서 Stage 1-A type filter 제거

### `test_output.txt` / `test_output_utf8.txt`

### Critical Bug #2: Metadata Truncation- 3가지 쿼리 종합 테스트 결과

- **발견**: `test_context_building.py`로 context 검증 중 발견- Argentina education, Stephen Graham, Bee Cliff 모두 성공 확인

- **문제**: Passage metadata를 150-200자로 잘라서 LLM에 전달 → 중요 정보 손실

- **예시**: "Dr. Alberto Taquini" 정보가 잘린 description에서 누락## 개발 결과

- **해결**: `sequential_answering.py`와 `query_decomposition.py`에서 전체 metadata 전달

### 이전 (단일 타입)

### Enhancement: Previous Context with Passages- Type matching: **0개**

- **발견**: SQ2가 SQ1의 답만 보고 passage는 못 봐서 정보 부족- 정답 발견: **실패**

- **해결**: `build_context_from_previous()`에서 이전 SQ의 retrieved passages도 포함

### 현재 (다중 타입)

## 📈 개발 과정 요약- Type matching: **23개** (3가지 타입 시도)

- 정답 발견: **성공** (Education in Argentina #1위)

### Phase 1: 다중 타입 추출 시스템 (초기)- Tried types: `EducationalSystem` → `SocialSystem` → `AcademicField`

- **문제**: LLM이 추출한 단일 타입이 DB 스키마와 불일치

- **해결**: 엔티티당 2-3개 possible_types 추출## 주요 코드 변경

- **결과**: Type matching 0개 → 23개로 증가

1. **Prompt/entity_extraction_prompt.py**

### Phase 2: Type Filter Bug 수정   - Output: `type/subtype` → `possible_types` 배열

- **문제**: Stage 1-A에서 타입 필터링으로 cross-type 매칭 실패   - 각 엔티티당 2-3개 타입 조합 제공

- **해결**: Stage 1-A를 pure value-based search로 변경

- **결과**: Argentina 케이스 13개 results 검색 성공2. **hybrid_retrieval.py**

   - `stage1b_type_filtering()`: 모든 타입 조합 시도

### Phase 3: Metadata Truncation 수정   - `retrieve_for_entity_hybrid()`: possible_types 추출 및 전달

- **문제**: Passage 정보 150-200자로 잘라 중요 정보 손실

- **해결**: 전체 metadata 전달## 보관 이유

- **결과**: Answer quality 개선

향후 참고를 위해 개발 과정과 검증 방법을 보존합니다.

### Phase 4: Context Enhancement실제 사용 테스트는 `test_hybrid_retrieval.py` 사용 권장.

- **문제**: Dependent SQ가 이전 passages 못 봄
- **해결**: Previous context에 passages 포함
- **결과**: Multi-hop reasoning 능력 향상

## 🏆 최종 성능 (200 Questions)

### Answer Quality
- Exact Match: 39.5%
- F1 Score: 50.3%
- Accuracy: 47.5%

### Retrieval Performance
- Macro Recall: 90.2%
- Perfect Recall Rate: 82.5%
- Bridge: 89.2% / Comparison: 94.0%

## 💡 보관 이유

향후 참고를 위해 개발 과정과 검증 방법을 보존합니다.
실제 평가는 `test_multihop_200.py` 및 `evaluate_retrieval_recall.py` 사용 권장.
