# Perfect Recall but Low Accuracy 분석

## 🎯 문제 정의
- **Perfect Recall**: 모든 supporting facts를 성공적으로 검색함 (Recall = 1.0)
- **Low Accuracy**: 그럼에도 최종 답변이 정답과 일치하지 않음

## 📊 통계
- Total Questions: **200**
- Perfect Recall Cases: **179/200 (89.5%)**
- Perfect Recall + Low Accuracy: **62/179 (34.6%)**

이는 **모든 필요한 정보를 찾았음에도 34.6%가 틀린 답을 생성**함을 의미합니다.

## 🔍 주요 실패 패턴

### Pattern 1: Format Mismatch (15/62 cases, 24.2%)
정답의 핵심 내용은 맞지만 형식이 다름

**예시:**
- Gold: `"78.5 mi long"` → Pred: `"78.5 miles"` ❌
- Gold: `"directed by Shane Meadows"` → Pred: `"Shane Meadows"` ❌
- Gold: `"Wilhelmus Simon Petrus Fortuijn, known as Pim Fortuyn"` → Pred: `"Pim Fortuyn"` ❌

**원인**: Accuracy 메트릭이 부분 문자열 포함(`gold in pred`)만 체크하므로 형식 차이에 민감

### Pattern 2: Completely Wrong Answer (42/62 cases, 67.7%)
정답과 완전히 다른 답변 (< 30% word overlap)

---

## 🚨 핵심 문제: Sub-Question 답변 실패의 연쇄 효과

### Case Study 1: Cher 질문
**Question**: "In which year was this single by Cher, written by Brian Higgins and included in the album The Very Best of Cher, released?"

**Gold Answer**: `1998`  
**Predicted Answer**: `2003` ❌

**Retrieved Supporting Facts**: ✅ All found
- "The Very Best of Cher" ✓
- "Believe (Cher song)" ✓

**Sub-Question 분석**:
```
SQ1: What is the title of the single by Cher that was written by Brian Higgins and included in the album The Very Best of Cher?
Answer: "Insufficient information." ❌

SQ2: In which year was [SQ1_Answer] released?
Answer: "2003" ❌
```

**문제**:
1. **SQ1 실패**: "Believe (Cher song)" 문서를 찾았음에도 곡 제목을 추출하지 못함
2. **연쇄 실패**: SQ1이 "Insufficient information"이므로 SQ2가 잘못된 전제로 추론
3. **Wrong Answer**: "The Very Best of Cher" 앨범의 발매년도 2003을 답변함

**Retrieved Passages for SQ1**:
- Believe (Cher song) ✓
- The Very Best of Cher ✓
- Rihanna (노이즈)
- Vada Nobles (노이즈)

---

### Case Study 2: Baltic Cup 질문
**Question**: "Which country refrained from participating in the 1991 Baltic Cup though it had participated in previous Baltic Cup competitions?"

**Gold Answer**: `Belarus`  
**Predicted Answer**: `Estonia` ❌

**Retrieved Supporting Facts**: ✅ All found
- "Baltic Cup (football)" ✓
- "Estonia national football team 1991" ✓

**Sub-Question 분석**:
```
SQ1: Which countries participated in previous Baltic Cup competitions?
Answer: "Insufficient information." ❌

SQ2: Which country from [SQ1_Answer] refrained from participating in the 1991 Baltic Cup?
Answer: "Estonia" ❌
```

**문제**:
1. **SQ1 실패**: "Baltic Cup (football)" 문서에 참가국 정보가 있음에도 추출 실패
2. **연쇄 실패**: SQ1이 실패하여 SQ2가 잘못된 추론
3. **Wrong Answer**: "Estonia national football team 1991" 문서를 보고 Estonia를 선택 (틀림)

**Retrieved Passages for SQ1**:
- Baltic Cup (football) ✓
- 1991 Baltic Cup ✓
- 1992 Baltic Cup
- 1995 Baltic Cup

---

## 🎯 근본 원인 분석

### 1. **LLM이 "Insufficient information" 남발**
- 모든 필요한 문서를 찾았음에도 답변 추출 실패
- 가능한 원인:
  - 프롬프트가 너무 보수적 (확실하지 않으면 "Insufficient"로 응답)
  - Passage에서 정보 추출 능력 부족
  - Metadata 정보는 있지만 핵심 답변이 없는 경우

### 2. **연쇄 실패 (Cascading Failure)**
- SQ1이 실패하면 → SQ2는 잘못된 전제로 추론
- `[SQ1_Answer]`가 "Insufficient information"이면 SQ2는 의미 없는 질문
- 하지만 LLM은 여전히 "최선의 추측"으로 답변 시도
- 이로 인해 **완전히 틀린 답변** 생성

### 3. **Prompt Engineering 문제**
현재 Sub-Question Answering Prompt:
- Final SQ용 짧은 프롬프트: max_tokens=100
- Intermediate SQ용 상세 프롬프트: max_tokens=200
- 하지만 둘 다 정보가 부족할 때 "Insufficient information" 반환 경향

---

## 💡 개선 방안

### 즉시 적용 가능:

#### 1. **Sub-Question Answering Prompt 개선**
```python
# 현재 (보수적):
"If the passages do not contain enough information, respond with 'Insufficient information.'"

# 개선안 (적극적):
"Extract the most relevant information from the passages. 
If direct answer is not found but related information exists, 
provide the closest match with confidence level."
```

#### 2. **Passage Ranking 개선**
- Top-K passages에 노이즈가 많음 (Rihanna, Vada Nobles 등)
- 관련도 점수로 필터링 강화
- LLM Filtering을 더 엄격하게

#### 3. **연쇄 실패 방지 메커니즘**
```python
# SQ1이 "Insufficient information"이면:
- Option A: Retrieval 재시도 (다른 키워드로)
- Option B: Query Decomposition 재시도 (다른 방식으로)
- Option C: Main Query에서 직접 답변 시도 (Decomposition 스킵)
```

### 중기 개선:

#### 4. **Answer Validation Layer**
- Sub-Question 답변 검증 단계 추가
- "Insufficient information" 감지 시 자동 재시도
- 답변 신뢰도 점수 계산

#### 5. **Hybrid Answering Strategy**
- Sequential (현재) + Direct (Main Query 직접 답변)
- 두 답변 비교 후 더 신뢰도 높은 것 선택

#### 6. **Few-shot Examples in Prompt**
- "Insufficient information" 대신 적극적으로 정보 추출하는 예시 추가

---

## 📈 기대 효과

현재:
- Perfect Recall: **179/200 (89.5%)**
- Perfect Recall + Correct Answer: **117/179 (65.4%)**

개선 후 목표:
- Perfect Recall → Correct Answer: **>80%**
- 현재 62개의 실패 케이스 중 최소 30개 이상 복구 가능

---

## 🔬 추가 분석 필요

1. **"Insufficient information" 발생 빈도**
   - 전체 Sub-Question 중 몇 %가 이 답변을 생성하는지
   - 어떤 유형의 질문에서 주로 발생하는지

2. **Passage Quality 분석**
   - Perfect Recall이지만 실패한 케이스의 Retrieved Passages 품질
   - Top-K에서 정답 포함 Passage의 순위

3. **Token F1 분포**
   - Perfect Recall 케이스의 Token F1 분포
   - Low Accuracy 케이스의 Token F1이 특히 낮은지

---

## ✅ Action Items

**Priority 1 (Immediate)**:
- [ ] Sub-Question Answering Prompt 개선 (더 적극적으로)
- [ ] "Insufficient information" 케이스 로깅 및 분석
- [ ] Few-shot examples 추가

**Priority 2 (Short-term)**:
- [ ] Answer Validation Layer 구현
- [ ] 연쇄 실패 방지 메커니즘 (재시도 로직)
- [ ] Passage Ranking 개선

**Priority 3 (Medium-term)**:
- [ ] Hybrid Answering Strategy (Sequential + Direct)
- [ ] Confidence scoring 시스템
- [ ] 자동 재시도 및 fallback 전략

