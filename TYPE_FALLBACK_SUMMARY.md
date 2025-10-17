# Type Filtering Fallback - Implementation Summary

## 🎯 문제 상황

**Query:** "What Cantonese slang term can mean both 'ghost man' and to refer to Westerners?"

### 문제:
- LLM이 `'ghost man'` (description)을 entity로 추출
- Type: `Concept/SocialSystem` 
- Metadata에 "ghost man"이 있는 passage: `Ghosts (2006 film)` 
- 하지만 이 passage의 type: `WorkOfArt/Film`
- **Type mismatch → 0 results**

### 근본 원인:
LLM이 query에서 entity의 type을 추론하는데, 실제 metadata의 type과 다를 수 있음

---

## ✅ 해결 방법: Option 1 - Type Filtering Fallback

### 전략:
1. **1차 시도**: Type 필터링으로 검색
2. **결과가 0개면**: Type 없이 재검색 (fallback)

### 구현 위치:
- `metadata_db.py`
  - `search_by_entity()` - LIKE 검색
  - `search_by_entity_fts()` - FTS 검색

### 코드 변경:

```python
# BEFORE (Type mismatch → 0 results)
def search_by_entity_fts(entity_name, entity_type, entity_subtype):
    query = "SELECT ... WHERE MATCH ? AND type = ? AND subtype = ?"
    return results  # 0 if type doesn't match

# AFTER (Type mismatch → fallback to no type filter)
def search_by_entity_fts(entity_name, entity_type, entity_subtype):
    # 1차: Type 필터링
    query = "SELECT ... WHERE MATCH ? AND type = ? AND subtype = ?"
    results = execute(query)
    
    # Fallback: 0개면 type 없이 재검색
    if len(results) == 0 and entity_type:
        query = "SELECT ... WHERE MATCH ?"
        results = execute(query)
    
    return results
```

---

## 📊 효과

### Before Fix:
```
Query: "What Cantonese slang term..."
Entity: 'ghost man' (Concept/SocialSystem)
Results: 0 passages  ❌
```

### After Fix:
```
Query: "What Cantonese slang term..."
Entity: 'ghost man' (Concept/SocialSystem)
1차 시도 (with type): 0 passages
Fallback (no type): 1 passage  ✅
  - Ghosts (2006 film)
```

---

## 🔍 Entity Extraction Prompt 확인

### 현재 Prompt 특징:
✅ **순수하게 query만 해석**
- "Use the exact name as it appears in the question"
- "Extract from the following question" (내부 지식 사용 X)

✅ **Type 추론은 query 기반**
- Question에서 context로 type 판단
- 예: "Cantonese slang term" → Concept/SocialSystem

### 한계:
- Query의 설명(description)과 metadata의 실제 entity 이름이 다를 수 있음
- 예: Query에 "ghost man" (설명) → 정답은 "Gweilo" (entity 이름)

---

## 🚀 추후 개선 방안 (Option 2 - 나중에 구현)

### 더 고급 Fallback 전략:

```python
def retrieve_with_advanced_fallback(entity):
    # 1차: Type 필터링
    results = search(entity, type, subtype)
    if results: return results
    
    # 2차: Type 없이 검색
    results = search(entity)
    if results: return results
    
    # 3차: Semantic search (embedding)
    results = semantic_search(entity)
    if results: return results
    
    # 4차: Relation traversal
    results = search_by_relation(entity)
    return results
```

---

## 📝 변경 파일

1. **metadata_db.py**
   - `search_by_entity()`: LIKE 검색 fallback 추가
   - `search_by_entity_fts()`: FTS 검색 fallback 추가

2. **테스트 파일**
   - `test_ghost_fix.py`: Ghost man query 검증
   - `test_quick_20.py`: 20개 query 빠른 테스트

---

## ✨ 결론

**Type 필터링을 완화**하여, type이 정확히 맞지 않아도 관련 passage를 찾을 수 있도록 개선했습니다.

- ✅ Type이 맞으면 → 정확한 결과
- ✅ Type이 안 맞으면 → Fallback으로 넓게 검색
- ✅ No-match 비율 감소 예상

다음 전체 200개 테스트에서 개선 효과를 확인할 수 있습니다!
