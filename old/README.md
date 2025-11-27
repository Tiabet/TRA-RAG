# Prompt Module Organization

이 폴더는 프로젝트 전체에서 사용되는 LLM 프롬프트를 관리합니다.

## 📁 파일 구조

```
Prompt/
├── entity_extraction_prompt.py     # 질문에서 엔티티 추출 (Role + Importance + Multiple Types)
├── title_filtering_prompt.py       # Title + Snippet 기반 LLM 필터링 (Stage 1-A, 1-B 공통)
├── metadata_construction_prompt.py # Passage에서 메타데이터 구조화 (DB 구축용)
├── type_schema.py                  # 엔티티 타입 스키마 정의
└── llm_filtering_prompt.py         # (Legacy) 사용 안 함
```

---

## 🔍 각 프롬프트 설명

### 1. **entity_extraction_prompt.py**
**용도**: 질문에서 엔티티 추출 (Hybrid Retrieval의 첫 단계)

**호출 위치**: 
- `hybrid_retrieval.py` → `extract_entities_from_query()`

**입력**: 
```python
"Stephen Graham starred in a film in 2006, directed by whom?"
```

**출력**:
```json
{
  "entities": [
    {
      "entity_name": "Stephen Graham",
      "possible_types": [
        {"type": "Person", "subtype": "Actor"},
        {"type": "Person", "subtype": "Artist"}
      ],
      "role": "target",
      "importance": "critical"
    },
    {
      "entity_name": "2006",
      "possible_types": [
        {"type": "Concept", "subtype": "TimePoint"},
        {"type": "Concept", "subtype": "Year"}
      ],
      "role": "attribute",
      "importance": "important"
    }
  ]
}
```

**특징**:
- ✅ Multiple Types: 엔티티당 2-3개 타입 후보 제공
- ✅ Role 분류: target / attribute / context
- ✅ Importance: critical / important / optional
- ✅ Query-only extraction: 질문에 명시된 것만 추출 (추론 금지!)

---

### 2. **title_filtering_prompt.py** ⭐ NEW!
**용도**: Candidate passages를 LLM으로 필터링 (Stage 1-A & 1-B 공통)

**호출 위치**:
- `hybrid_retrieval.py` → `stage1a_value_matching()` (Value/FTS 검색 후)
- `hybrid_retrieval.py` → `stage1b_type_filtering()` (Type DB 검색 후)

**입력**:
```json
{
  "query": "Stephen Graham starred in a film in 2006, directed by whom?",
  "entity_name": "Stephen Graham",
  "entity_type": "Person",
  "entity_subtype": "Actor",
  "candidates": [
    {
      "title": "Stephen Graham",
      "type": "Person",
      "subtype": "Actor",
      "snippet": "description: British actor known for This Is England (2006); main_entity: Actor active 1990-present"
    },
    {
      "title": "Stephen Wade",
      "type": "Person",
      "subtype": "Athlete",
      "snippet": "description: American swimmer, Olympic gold medalist; main_entity: Athlete specializing in swimming"
    }
  ]
}
```

**출력**:
```json
{
  "relevant_titles": ["Stephen Graham"],
  "filtered_out_titles": ["Stephen Wade"],
  "reasoning": "Stephen Graham matches - British actor in 2006 film. Stephen Wade filtered - swimmer, not actor, no 2006 film connection."
}
```

**특징**:
- ✅ Snippet 활용: Title + Type + Snippet으로 정확한 판단
- ✅ 0개 반환 가능: 전부 무관하면 빈 배열 반환
- ✅ Alias/Spelling 고려: "NHL" = "National Hockey League"
- ✅ Be INCLUSIVE: 애매하면 keep (Recall 보호)

---

### 3. **metadata_construction_prompt.py**
**용도**: Passage에서 구조화된 메타데이터 추출 (DB 구축 시 사용)

**호출 위치**:
- 현재는 사용 안 함 (DB가 이미 구축됨)
- DB 재구축 시 사용 예정

**입력**: Raw passage text

**출력**:
```json
{
  "main_entity": {
    "name": "Stephen Graham",
    "type": "Person",
    "subtype": "Actor"
  },
  "description": "British actor...",
  "attributes": {...},
  "events": {...}
}
```

---

### 4. **type_schema.py**
**용도**: 엔티티 타입/서브타입 스키마 정의

**사용 위치**:
- `entity_extraction_prompt.py`에서 import
- LLM이 올바른 타입 선택하도록 가이드

**내용**:
```python
ENTITY_TYPE_SCHEMA = """
Person: Actor, Writer, Athlete, Politician, ...
Location: Country, City, State, Airport, ...
Event: SportsTournament, Festival, Competition, ...
WorkOfArt: Film, Book, Album, ...
Organization: Company, SportsTeam, Institution, ...
Concept: SocialSystem, Genre, TimePoint, ...
Product: Vehicle, Software, ...
"""
```

---

### 5. **llm_filtering_prompt.py** (Legacy)
**상태**: ❌ 사용 안 함

**이유**: 
- 더 상세한 버전이었지만 `title_filtering_prompt.py`로 단순화
- Snippet + 0개 반환 로직만 추출해서 새 버전 제작

**보관 이유**: 
- 향후 참고용 (예시가 더 풍부함)
- 삭제 예정

---

## 🔄 프롬프트 사용 흐름

### Hybrid Retrieval Pipeline

```
Query
  ↓
┌────────────────────────────────────────────────┐
│ entity_extraction_prompt.py                    │
│ → 엔티티 추출 (Multiple Types + Role)          │
└────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────┐
│ Stage 1-A: Value/FTS Matching                  │
│ → DB 검색 (FTS)                                 │
│ → title_filtering_prompt.py (LLM Filter) ⭐    │
└────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────┐
│ Stage 1-B: Type Filtering                      │
│ → DB 검색 (Type/Subtype)                        │
│ → title_filtering_prompt.py (LLM Filter) ⭐    │
└────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────┐
│ Stage 2: Merge & Deduplicate                   │
└────────────────────────────────────────────────┘
```

---

## 📦 Import 방법

### hybrid_retrieval.py에서

```python
from Prompt.entity_extraction_prompt import ENTITY_EXTRACTION_PROMPT
from Prompt.title_filtering_prompt import TITLE_FILTERING_PROMPT

# 사용
prompt = ENTITY_EXTRACTION_PROMPT.replace("__QUESTION__", query)
prompt = TITLE_FILTERING_PROMPT.replace('{{QUERY}}', query)
```

---

## ✅ 장점

### Before (프롬프트 inline)
```python
# hybrid_retrieval.py (600+ lines)
TITLE_FILTERING_PROMPT = """...(100 lines)..."""

def stage1a_value_matching(...):
    prompt = TITLE_FILTERING_PROMPT.replace(...)
```

**문제점:**
- ❌ 가독성 최악 (600+ 줄 파일에 100줄 프롬프트)
- ❌ 유지보수 어려움 (프롬프트 찾기 힘듦)
- ❌ 재사용 불가 (Stage 1-A, 1-B에서 중복)

---

### After (프롬프트 분리)
```python
# hybrid_retrieval.py (500 lines, 20% 감소!)
from Prompt.title_filtering_prompt import TITLE_FILTERING_PROMPT

def stage1a_value_matching(...):
    prompt = TITLE_FILTERING_PROMPT.replace(...)
```

**장점:**
- ✅ 가독성 향상 (코드와 프롬프트 분리)
- ✅ 유지보수 쉬움 (Prompt/ 폴더에 모아두기)
- ✅ 재사용 가능 (Stage 1-A, 1-B 공통)
- ✅ 버전 관리 용이 (프롬프트 변경 추적)

---

## 🚀 향후 계획

1. **llm_filtering_prompt.py 삭제**
   - title_filtering_prompt.py로 완전 대체됨

2. **metadata_construction_prompt.py 업데이트**
   - DB 재구축 시 Multiple Types 지원 추가

3. **프롬프트 버전 관리**
   - 각 파일에 version, last_updated 추가
   - 성능 변화 추적용

4. **단위 테스트 추가**
   - 각 프롬프트별 예시 케이스 테스트
   - 프롬프트 변경 시 자동 검증
