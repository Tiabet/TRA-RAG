# LLM Filtering Input Length 문제 분석

## 🚨 발견된 문제

### 현재 상황
- **제한 없음**: LLM Filtering 시 모든 candidates를 그대로 전송
- **최대 케이스**: 1014개 candidates (full metadata 포함)
- **예상 문제**:
  - Input token limit 초과 가능성
  - API 비용 증가
  - 응답 시간 증가
  - LLM의 판단 품질 저하 (너무 많은 선택지)

### 발견된 케이스들
```
count: 1014  ← 최악의 케이스!
count: 723
count: 551
count: 472
count: 419
count: 418
count: 400
count: 396
count: 393
...
```

---

## 🔍 현재 코드 분석

### Stage 1-A (Value Matching)
```python
# hybrid_retrieval.py Line ~190
candidates = []
for m in matches:
    # Extract ALL metadata fields (no limit!)
    snippet_parts = []
    for key, value in metadata.items():
        if key not in excluded_keys and value:
            value_str = str(value)  # FULL value!
            snippet_parts.append(f"{key}: {value_str}")
    
    candidates.append({...})

# Send ALL candidates to LLM (no limit!)
prompt = prompt.replace('{{CANDIDATES}}', json.dumps(candidates, indent=2, ensure_ascii=False))
```

**문제**: 1000개 × full metadata = 수십만 토큰 가능

### Stage 1-B (Type Filtering)
```python
# hybrid_retrieval.py Line ~340
# Same issue - no limit on candidates
prompt = prompt.replace('{{CANDIDATES}}', json.dumps(candidates, indent=2, ensure_ascii=False))
```

---

## 💡 해결 방안

### Option 1: Top-K Limit (Simple, Fast)
```python
MAX_CANDIDATES_FOR_LLM = 50  # Reasonable limit

if len(candidates) > MAX_CANDIDATES_FOR_LLM:
    # Take top-K by some criteria
    # Option A: Random sample
    # Option B: First K (assuming some pre-ranking)
    # Option C: Stratified sampling
    candidates = candidates[:MAX_CANDIDATES_FOR_LLM]
    filter_info['truncated'] = True
    filter_info['original_count'] = original_count
```

**장점**:
- 간단하고 빠름
- Token 사용량 예측 가능

**단점**:
- 관련 passage를 놓칠 수 있음

### Option 2: Hierarchical Filtering (Better Quality)
```python
if len(candidates) > 50:
    # Stage 1: Quick keyword-based filtering
    keywords = extract_keywords(query, entity_name)
    candidates = keyword_filter(candidates, keywords, top_k=100)
    
    if len(candidates) > 50:
        # Stage 2: Batch LLM filtering
        # Split into batches of 50
        all_relevant = []
        for batch in batches(candidates, size=50):
            relevant = await llm_filter(batch)
            all_relevant.extend(relevant)
        return all_relevant
```

**장점**:
- 품질 유지
- 모든 candidates 검토 가능

**단점**:
- 복잡함
- 여러 LLM 호출 (비용↑, 시간↑)

### Option 3: Metadata Truncation (Current + Limit)
```python
MAX_METADATA_LENGTH = 200  # characters per field

for key, value in metadata.items():
    value_str = str(value)
    if len(value_str) > MAX_METADATA_LENGTH:
        value_str = value_str[:MAX_METADATA_LENGTH] + "..."
    snippet_parts.append(f"{key}: {value_str}")
```

**장점**:
- Token 제어 가능
- 여전히 많은 candidates 처리 가능

**단점**:
- 정보 손실

### Option 4: Hybrid Approach (추천!)
```python
MAX_CANDIDATES_FOR_LLM = 100
MAX_METADATA_LENGTH = 300

# 1. Limit candidate count
if len(candidates) > MAX_CANDIDATES_FOR_LLM:
    # Keyword pre-filtering
    candidates = keyword_prefilter(candidates, query, top_k=MAX_CANDIDATES_FOR_LLM)
    filter_info['prefiltered'] = True

# 2. Truncate metadata for each candidate
for candidate in candidates:
    for key, value in metadata.items():
        value_str = str(value)
        if len(value_str) > MAX_METADATA_LENGTH:
            value_str = value_str[:MAX_METADATA_LENGTH] + "..."
        ...
```

**장점**:
- Token 사용량 제어
- 품질 유지 (키워드 pre-filter)
- 구현 중간 난이도

**단점**:
- 일부 정보 손실

---

## 📊 예상 Token 사용량

### 현재 (제한 없음)
```
1000 candidates × 500 tokens/candidate = 500,000 tokens
→ gpt-4o-mini input limit: 128,000 tokens
→ EXCEEDED! 오류 발생 가능
```

### Option 4 적용 시
```
100 candidates × 300 tokens/candidate = 30,000 tokens
→ Safe! + Reasonable cost
```

---

## 🎯 권장 사항

### Immediate (High Priority)
1. **MAX_CANDIDATES_FOR_LLM = 50** 적용
2. **Keyword pre-filtering** 구현
   - Query + Entity name에서 키워드 추출
   - Metadata에 키워드 포함된 것 우선
3. **Metadata length truncation** (선택적)

### Short-term (Medium Priority)
4. **Logging 추가**: 몇 개가 truncate 되었는지 추적
5. **Error handling**: Token limit 초과 시 재시도 로직

### Long-term (Low Priority)
6. **Hierarchical filtering** 실험
7. **Adaptive limit**: Query 복잡도에 따라 동적 조정

---

## 📝 구현 예시

```python
# hybrid_retrieval.py

MAX_CANDIDATES_FOR_LLM = 50
MAX_METADATA_FIELD_LENGTH = 300

async def stage1a_value_matching(...):
    # ... existing code ...
    
    initial_count = len(matches)
    
    if apply_llm_filter and matches:
        # Prepare candidates
        candidates = []
        for m in matches:
            metadata_str = m.get('metadata', '')
            # ... parse metadata ...
            
            # Build snippet with TRUNCATION
            snippet_parts = []
            excluded_keys = {'title', 'type', 'subtype'}
            
            for key, value in metadata.items():
                if key not in excluded_keys and value:
                    value_str = str(value)
                    # TRUNCATE long fields
                    if len(value_str) > MAX_METADATA_FIELD_LENGTH:
                        value_str = value_str[:MAX_METADATA_FIELD_LENGTH] + "..."
                    snippet_parts.append(f"{key}: {value_str}")
            
            snippet = '\n'.join(snippet_parts) if snippet_parts else 'No metadata'
            
            candidates.append({
                'title': m['title'],
                'type': m.get('type', 'Unknown'),
                'subtype': m.get('subtype', 'Unknown'),
                'matched_content': snippet
            })
        
        # LIMIT candidate count
        truncated = False
        original_count = len(candidates)
        
        if len(candidates) > MAX_CANDIDATES_FOR_LLM:
            # Option A: Simple truncation
            candidates = candidates[:MAX_CANDIDATES_FOR_LLM]
            truncated = True
            
            # TODO Option B: Keyword pre-filtering (better!)
            # candidates = keyword_prefilter(candidates, query, entity_name, MAX_CANDIDATES_FOR_LLM)
        
        # Format prompt
        prompt = TITLE_FILTERING_PROMPT.replace(...)
        prompt = prompt.replace('{{CANDIDATES}}', json.dumps(candidates, indent=2, ensure_ascii=False))
        
        # ... LLM call ...
        
        filter_info = {
            'stage': '1-A',
            'initial_matches': initial_count,
            'llm_candidates': len(candidates),
            'llm_filtered': len(filtered_matches),
            'truncated': truncated,
            'original_candidate_count': original_count if truncated else len(candidates),
            'reasoning': result.get('reasoning', '')
        }
```

---

## ✅ Action Items

- [ ] Implement MAX_CANDIDATES_FOR_LLM = 50
- [ ] Add metadata field truncation (300 chars)
- [ ] Add logging for truncation events
- [ ] (Optional) Implement keyword pre-filtering
- [ ] Test with 1014-candidate case
- [ ] Monitor token usage improvement

