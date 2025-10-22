# LLM Filtering Improvements

**날짜**: 2025-10-20  
**변경 사항**: Snippet 추가 + 0개 반환 명시적 허용

---

## 📊 변경 전후 비교

### Before (Title만 사용)

```python
# Stage 1-A & 1-B 모두
candidate_titles = [c['title'] for c in candidates]

# LLM에 전송되는 데이터
{
  "candidates": [
    "Stephen Graham",
    "This Is England",
    "Shane Meadows"
  ]
}
```

**문제점:**
- ❌ Title만으로는 판단 어려움 (동명이인, 동명 이벤트)
- ❌ Metadata 정보 활용 불가
- ❌ False positive 많음

---

### After (Title + Snippet)

```python
# Stage 1-A & 1-B 모두
candidates_with_snippets = []
for c in candidates:
    metadata = json.loads(c.get('metadata', '{}'))
    
    # Key fields에서 snippet 생성
    snippet_parts = []
    for key in ['description', 'main_entity', 'attributes', 'events']:
        if key in metadata:
            snippet_parts.append(f"{key}: {metadata[key][:100]}")
    
    snippet = '; '.join(snippet_parts[:2])[:150]  # 최대 2개 필드, 150자
    
    candidates_with_snippets.append({
        'title': c['title'],
        'type': c.get('type'),
        'subtype': c.get('subtype'),
        'snippet': snippet
    })

# LLM에 전송되는 데이터
{
  "candidates": [
    {
      "title": "Stephen Graham",
      "type": "Person",
      "subtype": "Actor",
      "snippet": "description: British actor known for This Is England (2006), Snatch (2000); main_entity: Actor active 1990-present"
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

**개선점:**
- ✅ Snippet으로 정확한 판단 가능 (Actor vs Athlete)
- ✅ Metadata의 description, main_entity, attributes, events 활용
- ✅ False positive 대폭 감소 예상

---

## 🎯 프롬프트 개선

### 1. **0개 반환 명시적 허용**

**Before:**
```
**IMPORTANT**: Be INCLUSIVE rather than exclusive. If there's reasonable chance...
```

**After:**
```
**IMPORTANT RULES:**
- Be INCLUSIVE rather than exclusive - when in doubt, keep it
- **If ALL candidates are irrelevant, return EMPTY ARRAY []** - don't force-keep any
- Use both title AND snippet to make decisions

...

**ZERO RESULTS OK:** If no candidates are relevant, return `"relevant_titles": []`
```

**효과:**
- ✅ LLM이 억지로 1개 선택하지 않음
- ✅ 완전히 상관없는 케이스에서 0개 반환 가능
- ✅ Precision 향상 (잘못된 passage 제거)

---

### 2. **Alias/Spelling Variation 명시**

**추가된 규칙:**
```
3. **Alias/spelling variations**: Consider "NHL" = "National Hockey League", 
   "Roissy" = "Charles de Gaulle Airport"
```

**예시 추가:**
```
- Query: "ghost man slang" → Keep: "Gweilo" if snippet says "Cantonese slang for ghost man"
- Query: "Baltic Cup" → Remove: "FIFA World Cup" (same type, completely different event)
```

---

## 📈 예상 효과

### Precision 향상

| Stage | Before | After (예상) | 개선 |
|-------|--------|-------------|------|
| 1-A (Value) | 3.62% | **8-12%** | 2-3배 ↑ |
| 1-B (Type) | 12.09% | **15-18%** | 1.3배 ↑ |

**근거:**
- Snippet으로 동명이인/동명 이벤트 구분 가능
- 0개 반환으로 완전히 상관없는 케이스 제거
- Metadata의 description/main_entity가 핵심 정보 제공

---

### Recall 영향

**우려:** Snippet 추가로 LLM이 너무 보수적으로 필터링?

**대응:**
- "Be INCLUSIVE" 규칙 유지
- "when in doubt, keep it" 명시
- 0개 반환은 **전부 상관없을 때만**

**예상:** Recall 변화 미미 (78.3% → 78-79%)

---

### 비용 영향

**Token 증가:**
```
Before: Title만 (평균 10 tokens/candidate)
After:  Title + Snippet (평균 50 tokens/candidate)

Stage 1-A: 16개 × 50 = 800 tokens/query
Stage 1-B: 109개 → 3.8개 필터링 후, 평균 20개 × 50 = 1000 tokens/query

총 증가: 1800 tokens/query (input)
```

**200 queries 기준:**
```
200 queries × 1800 tokens = 360K tokens 추가 (input)
gpt-4o-mini 가격: $0.15/1M tokens
추가 비용: $0.054 (약 5센트)
```

**결론:** 비용 증가 미미하지만 Precision 2-3배 향상! 🎉

---

## 🔍 Snippet 생성 로직

### 우선순위 필드 (순서대로 최대 2개)

1. **description** - 가장 중요! 엔티티 설명
2. **main_entity** - 핵심 엔티티 정보
3. **attributes** - 속성 (날짜, 위치, 통계 등)
4. **events** - 이벤트 정보 (시간순)

### 길이 제한

- 각 필드: 최대 100자
- 총 snippet: 최대 150자
- 최대 2개 필드만 포함

**예시:**
```json
{
  "title": "Stephen Graham",
  "snippet": "description: British actor known for This Is England (2006); main_entity: Actor born 1973, active in film and TV"
}
```

---

## 🧪 테스트 계획

### 1. **Quick Test (3 queries)**
```bash
python hybrid_retrieval.py
```

**확인사항:**
- Snippet이 제대로 생성되는가?
- LLM이 snippet을 활용하는가?
- 0개 반환하는 케이스 있는가?

### 2. **Full Test (200 queries)**
```bash
python test_hybrid_200.py
```

**비교 지표:**
| 지표 | Before | After | 목표 |
|------|--------|-------|------|
| Recall | 78.3% | ? | 78-79% (유지) |
| Stage 1-A Precision | 3.62% | ? | 8-12% (2-3배) |
| Stage 1-B Precision | 12.09% | ? | 15-18% (1.3배) |
| No Recall Cases | 7 | ? | 5-6 (감소) |

---

## 📝 코드 변경 요약

### 파일: `hybrid_retrieval.py`

1. **`TITLE_FILTERING_PROMPT` 개선**
   - Snippet 필드 추가
   - 0개 반환 명시적 허용
   - Alias/spelling variation 규칙 추가

2. **`stage1a_value_matching()` 함수**
   - Title만 → Title + Snippet
   - Metadata에서 key fields 추출
   - 최대 150자 snippet 생성

3. **`stage1b_type_filtering()` 함수**
   - 동일하게 Title + Snippet
   - 중복 제거 로직 유지

---

## 🚀 Next Steps

1. ✅ Quick test 실행 중 (3 queries)
2. ⏳ Full test 실행 예정 (200 queries)
3. ⏳ 결과 분석 및 비교
4. ⏳ No Recall 케이스 재분석

**예상 완료**: 2025-10-20 오후
