# SQ 답변 생성 시 Metadata 처리 및 Error 처리 분석

## 📊 현재 동작 방식

### 1. SQ 답변 생성에서 사용하는 데이터

**위치**: `sequential_answering.py` Line 215-260

```python
async def generate_answer_from_passages(...):
    # Format passages
    passage_texts = []
    for i, passage in enumerate(passages[:10], 1):  # Top 10 passages
        title = passage.get('title', 'Unknown')
        metadata = passage.get('metadata', {})
        
        # Build passage text
        passage_parts = [f"[{i}] {title}"]
        
        # Include ALL metadata fields (no truncation, no field selection)
        excluded_keys = {'title', 'type', 'subtype'}
        for key, value in metadata.items():
            if key not in excluded_keys and value:
                value_str = str(value)  # FULL value, no truncation!
                passage_parts.append(f"  {key}: {value_str}")
        
        passage_texts.append('\n'.join(passage_parts))
```

**결론**: ✅ **맞습니다! FULL metadata를 사용합니다.**
- **Top 10 passages만** 사용
- 각 passage의 **모든 metadata fields** 포함
- **값 truncation 없음** - `str(value)` 전체를 사용

---

## 🔄 데이터 흐름

### Stage 1: LLM Filtering (hybrid_retrieval.py)
```
Initial matches (예: 1014개)
    ↓
LLM Filtering (문제: 제한 없음!)
    ↓
Filtered passages (예: 285개)
    ↓
Return: Full passage objects with metadata
```

### Stage 2: Answer Generation (sequential_answering.py)
```
Filtered passages (예: 285개)
    ↓
Take Top 10 only
    ↓
Extract FULL metadata for each
    ↓
Format as text (NO truncation)
    ↓
Send to LLM for answer generation
```

---

## 🚨 문제점 분석

### Problem 1: LLM Filtering Input 초과
**발생 위치**: `hybrid_retrieval.py` - LLM Filtering

**시나리오**:
```
1014 candidates × 500 tokens/each = 507,000 tokens
→ Exceeds gpt-4o-mini limit (128,000 tokens)
→ API Error!
```

**현재 Error 처리** (Line ~240):
```python
except Exception as e:
    log_llm_error(...)
    # Falls back to using ALL initial matches (no filtering!)
    return matches, {
        'stage': '1-A',
        'initial_matches': initial_count,
        'llm_filtered': initial_count,  # Same as initial!
        'reasoning': f'Error: {str(e)}'
    }
```

**결과**: 
- ❌ LLM filtering 실패
- ⚠️ **모든 initial matches를 그대로 반환** (1014개 전부!)
- 다음 단계로 진행 (에러는 숨겨짐)

### Problem 2: Answer Generation Input 초과
**발생 위치**: `sequential_answering.py` - Answer Generation

**시나리오**:
```
Top 10 passages × 50KB metadata/each = 500KB text
→ ~100,000+ tokens
→ Might exceed limit!
```

**현재 Error 처리** (Line ~338):
```python
except Exception as e:
    log_llm_error(...)
    return f"Error generating answer: {str(e)}"
```

**결과**:
- ❌ Answer generation 실패
- ⚠️ **"Error generating answer: ..."** 문자열 반환
- 이 에러 메시지가 SQ의 답변이 됨!
- 다음 SQ가 이 에러를 전제로 추론 (연쇄 실패)

---

## 📈 실제 케이스 분석

### 케이스 1: 1014 candidates
```json
{
  "retrieved_passages": {
    "count": 1014,
    "titles": [...]
  }
}
```

**추정 흐름**:
1. LLM Filtering 시도
2. Token limit 초과 → **Exception**
3. Fallback: 1014개 전부 반환
4. Answer Generation: Top 10 사용
5. 각 passage의 full metadata 포함
6. Token limit 초과 가능성 있음!

### 케이스 2: 285 passages (Stephen Graham)
```json
{
  "retrieved_passages": {
    "count": 285,
    "titles": [...]
  }
}
```

**Line 35981** (사용자가 보고 있는 케이스)

이 케이스에서:
1. LLM Filtering 성공 (285개는 처리 가능)
2. Answer Generation: Top 10 사용
3. Full metadata 포함
4. 만약 각 passage의 metadata가 크다면?
   - Token limit 초과 가능!
   - "Error generating answer: ..." 반환

---

## 🎯 문제 요약

### LLM Filtering 단계
- **제한**: ❌ 없음!
- **Error 처리**: Fallback to all matches (필터링 없이 전부 반환)
- **문제**: 
  - 1000+ candidates 시 token limit 초과
  - 필터링 실패 시 모든 candidates가 다음 단계로 전달됨

### Answer Generation 단계
- **제한**: ✅ Top 10 passages만
- **Metadata**: ❌ Full metadata (truncation 없음)
- **Error 처리**: "Error generating answer: ..." 반환
- **문제**:
  - Top 10이지만 각 metadata가 매우 클 수 있음
  - 100KB+ metadata → token limit 초과
  - 에러 메시지가 답변이 되어 연쇄 실패

---

## 💡 해결 방안

### Immediate Fixes (High Priority)

#### 1. LLM Filtering - Candidate Limit
```python
MAX_CANDIDATES_FOR_LLM_FILTER = 50

if len(candidates) > MAX_CANDIDATES_FOR_LLM_FILTER:
    # Option A: Take first 50 (assuming some pre-ranking)
    candidates = candidates[:MAX_CANDIDATES_FOR_LLM_FILTER]
    
    # Option B: Random sample
    import random
    candidates = random.sample(candidates, MAX_CANDIDATES_FOR_LLM_FILTER)
    
    filter_info['truncated_for_llm'] = True
    filter_info['original_count'] = original_count
```

#### 2. LLM Filtering - Metadata Field Truncation
```python
MAX_METADATA_FIELD_LENGTH = 300  # chars per field

for key, value in metadata.items():
    value_str = str(value)
    if len(value_str) > MAX_METADATA_FIELD_LENGTH:
        value_str = value_str[:MAX_METADATA_FIELD_LENGTH] + "..."
    snippet_parts.append(f"{key}: {value_str}")
```

#### 3. Answer Generation - Metadata Truncation
```python
MAX_METADATA_FIELD_FOR_ANSWER = 500  # More generous for answer generation

for key, value in metadata.items():
    if key not in excluded_keys and value:
        value_str = str(value)
        if len(value_str) > MAX_METADATA_FIELD_FOR_ANSWER:
            value_str = value_str[:MAX_METADATA_FIELD_FOR_ANSWER] + "..."
        passage_parts.append(f"  {key}: {value_str}")
```

#### 4. Answer Generation - Token Estimation & Fallback
```python
# Estimate tokens
estimated_tokens = len(passages_text) / 4  # Rough estimate

if estimated_tokens > 100000:  # Conservative limit
    # Fallback: Truncate to fewer passages
    passages = passages[:5]  # Use only top 5
    # Rebuild passages_text...
    
    log_warning(
        "Large input detected, truncated to top 5 passages",
        {"estimated_tokens": estimated_tokens}
    )
```

#### 5. Better Error Handling
```python
except Exception as e:
    error_msg = str(e)
    
    # Check if it's a token limit error
    if "maximum context length" in error_msg.lower() or "tokens" in error_msg.lower():
        log_llm_error(...)
        
        # Retry with fewer passages
        if len(passages) > 5:
            return await generate_answer_from_passages(
                client, subquestion, passages[:5],
                previous_context, is_final_sq, main_query
            )
    
    return "Insufficient information."  # Better than error message
```

---

## 📊 예상 개선 효과

### Before
```
LLM Filtering:
  1014 candidates × 500 tokens = 507,000 tokens → ERROR
  Fallback: Return all 1014 passages

Answer Generation:
  Top 10 × 10KB metadata = 100KB text
  ~100,000 tokens → Might ERROR
  Return: "Error generating answer: ..."
```

### After
```
LLM Filtering:
  50 candidates × 300 tokens = 15,000 tokens → OK
  Filtered: Return ~10-20 relevant passages

Answer Generation:
  Top 10 × 500 chars/field = manageable
  ~20,000 tokens → OK
  Return: Actual answer
```

---

## ✅ Action Items

**Priority 1 (Critical)**:
- [ ] Add MAX_CANDIDATES_FOR_LLM_FILTER = 50
- [ ] Add metadata field truncation in LLM Filtering
- [ ] Add metadata field truncation in Answer Generation
- [ ] Improve error handling (retry with fewer passages)

**Priority 2 (Important)**:
- [ ] Add token estimation before LLM calls
- [ ] Add logging for truncation events
- [ ] Change error fallback from error message to "Insufficient information"

**Priority 3 (Nice to have)**:
- [ ] Implement adaptive limits based on metadata size
- [ ] Add metrics to track truncation frequency

