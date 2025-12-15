# Stage 1-A vs 1-B 중요도 비교 분석

## 📊 Executive Summary

**결론**: **Stage 1-A (Value/FTS)가 더 중요하지만, Stage 1-B (Type filtering)는 필수적인 보완재**

- **Stage 1-A 없이는**: Recall 75.7% → 69.7% (-8.0%p, 116 facts 손실)
- **Stage 1-B 없이는**: Recall 78.3% → 75.7% (-2.6%p, 93 facts 손실)
- **Hybrid 사용 시**: **78.3% Recall** (최고)

---

## 1. 전체 검색 결과 통계

### 검색량 비교

| Stage | 총 검색량 | 평균/query | 비고 |
|-------|----------|-----------|------|
| **Stage 1-A (Value/FTS)** | 3,202 passages | 16.01 | 직접 사용 가능 |
| **Stage 1-B (Type)** | 21,864 candidates | 109.32 | 필터링 전 |
| **Stage 1-B (LLM filtered)** | 769 passages | 3.84 | **96.5% 제거** |
| **Stage 2 (Final)** | 3,602 passages | 18.01 | Unique merge |

**인사이트**:
- Stage 1-B는 **109개 후보 → 3.8개로 압축** (96.5% 필터링)
- LLM filtering이 매우 공격적 (과도한 제거 가능성)
- Final merge 후 18개 = Stage 1-A(16) + Stage 1-B(3.8) - 중복

---

## 2. Recall 기여도 (핵심 지표!)

### Supporting Facts 검색 현황

```
총 필요: 489 facts
검색 성공: 383 facts (78.3%)

┌─────────────────────────────────────────────┐
│ Stage 1-A만:     116 facts (30.3%)          │  ← 1-A 없으면 못 찾음
│ Stage 1-B만:      93 facts (24.3%)          │  ← 1-B 없으면 못 찾음
│ Both (중복):     174 facts (45.4%)          │  ← 양쪽 다 찾음 (고신뢰)
│─────────────────────────────────────────────│
│ Stage 1-A 총:    290 facts (75.7%)          │  ← 1-A의 전체 기여
│ Stage 1-B 총:    267 facts (69.7%)          │  ← 1-B의 전체 기여
└─────────────────────────────────────────────┘
```

### 핵심 발견

1. **Stage 1-A가 6.0%p 더 기여** (75.7% vs 69.7%)
   - Value/FTS matching이 더 강력한 기본 엔진
   
2. **독립 기여도는 비슷** (30.3% vs 24.3%)
   - 1-A만: 116 facts (이름 정확 매칭)
   - 1-B만: 93 facts (Type 기반 발견)
   - → **각각 독자적 가치 보유**

3. **중복(Both)이 45.4%로 가장 많음**
   - 174 facts를 **양쪽 모두** 찾음
   - → 이들은 **high-confidence** passage (재랭킹 시 우선순위)

---

## 3. Query별 의존도 분석

**Recall 있는 193 queries 분석**:

| 의존 타입 | 쿼리 수 | 비율 | 의미 |
|----------|---------|------|------|
| **Stage 1-A만 의존** | 62 | 32.1% | 1-A 없으면 No Recall |
| **Stage 1-B만 의존** | 63 | 32.6% | 1-B 없으면 No Recall |
| **Both 필요** | 7 | 3.6% | 둘 다 있어야 Full Recall |
| **Both 중복** | 61 | 31.6% | 어느 쪽이든 OK |

### 인사이트

- **32%의 쿼리**는 Stage 1-A에 절대 의존
- **33%의 쿼리**는 Stage 1-B에 절대 의존
- → **양쪽 모두 필수적**, 어느 한쪽만으로는 불충분

**시사점**: 
- Hybrid 접근이 정당화됨
- 단일 Stage로는 **1/3의 쿼리가 실패**

---

## 4. 효율성 분석

### Precision (정밀도)

| Stage | Supporting Facts | 총 검색량 | Precision |
|-------|-----------------|----------|-----------|
| **Stage 1-A** | 116 + 174 = 290 | 3,202 | **9.06%** |
| **Stage 1-B** | 93 + 174 = 267 | 769 | **34.72%** |

**단독 기여 기준**:
| Stage | 독립 Facts | 검색량 | Precision |
|-------|-----------|--------|-----------|
| **Stage 1-A** | 116 | 3,202 | **3.62%** |
| **Stage 1-B** | 93 | 769 | **12.09%** |

### 노이즈 비율

| Stage | 노이즈 passages | 총 검색량 | 노이즈 비율 |
|-------|----------------|----------|-----------|
| **Stage 1-A** | 2,912 | 3,202 | **90.9%** |
| **Stage 1-B** | 502 | 769 | **65.3%** |

### LLM Filtering 효과

```
필터링 전:  21,864 candidates
필터링 후:     769 passages
제거율:       96.5%
```

**문제점**: 
- 너무 공격적인 필터링 (96.5% 제거)
- Supporting facts도 일부 손실 가능성
- 21,864개 LLM 평가 = 높은 비용/시간

---

## 5. 특성 비교표

| 항목 | Stage 1-A (Value/FTS) | Stage 1-B (Type + LLM) |
|------|---------------------|---------------------|
| **속도** | ⭐⭐⭐⭐⭐ 매우 빠름 (FTS index) | ⭐⭐ 느림 (DB + LLM) |
| **Recall 기여** | ⭐⭐⭐⭐⭐ 75.7% | ⭐⭐⭐⭐ 69.7% |
| **독립 기여** | ⭐⭐⭐⭐ 30.3% | ⭐⭐⭐ 24.3% |
| **Precision** | ⭐ 3.62% | ⭐⭐⭐ 12.09% (3.3배!) |
| **노이즈** | ⭐ 90.9% 높음 | ⭐⭐⭐ 65.3% 중간 |
| **비용** | ⭐⭐⭐⭐⭐ 무료 (DB query) | ⭐⭐ 높음 (LLM API) |
| **구현 복잡도** | ⭐⭐⭐⭐⭐ 간단 | ⭐⭐ 복잡 (multiple types, LLM) |

### Stage 1-A 장단점

✅ **장점**:
- 매우 빠른 속도 (FTS index 활용)
- 높은 Recall 기여 (75.7%)
- 엔티티 이름 정확 매칭에 강함
- 비용 없음
- 구현 간단

❌ **단점**:
- 매우 낮은 Precision (3.62%)
- 높은 노이즈 (90.9%)
- Type 정보 활용 못 함
- 의미적 매칭 불가

### Stage 1-B 장단점

✅ **장점**:
- **3.3배 높은 Precision** (12.09%)
- 낮은 노이즈 (65.3%)
- 의미적 필터링 (LLM semantic matching)
- Value로 못 찾는 것 보완 (24.3% 독립 기여)
- Type 정보 활용

❌ **단점**:
- 느린 속도 (Type query + LLM call)
- 과도한 후보 생성 (평균 109개)
- 높은 LLM 비용 (21,864 candidates 평가)
- 복잡한 구현 (multiple types system)
- 96.5% 제거 = 과도한 필터링 가능성

---

## 6. 시뮬레이션: Stage 단독 사용 시

### Scenario 1: Stage 1-A만 사용

**성능**:
- Recall: **75.7%** (현재 78.3% 대비 **-2.6%p**)
- 못 찾는 facts: **93개** (24.3%)
- Query 실패: **63개** (32.6%)

**장점**:
- ✅ 빠른 속도
- ✅ 구현 간단
- ✅ 비용 없음
- ✅ 여전히 75.7% Recall (나쁘지 않음)

**단점**:
- ❌ 노이즈 매우 높음 (90.9%)
- ❌ Downstream re-ranking 부담 증가
- ❌ Type 정보 미활용

**결론**: **Acceptable baseline, but sub-optimal**

---

### Scenario 2: Stage 1-B만 사용

**성능**:
- Recall: **69.7%** (현재 78.3% 대비 **-8.6%p**)
- 못 찾는 facts: **116개** (30.3%)
- Query 실패: **62개** (32.1%)

**장점**:
- ✅ 높은 Precision (12.09%)
- ✅ 낮은 노이즈 (65.3%)
- ✅ 의미적 필터링

**단점**:
- ❌ **더 낮은 Recall** (-8.6%p, 1-A보다 6.0%p 낮음)
- ❌ 느린 속도
- ❌ 높은 LLM 비용
- ❌ 엔티티 이름 정확 매칭 약함

**결론**: **Not recommended as standalone**

---

### Scenario 3: Hybrid (현재)

**성능**:
- Recall: **78.3%** (최고)
- Both contribution: **45.4%** (high-confidence)
- Query 성공: **193/200** (96.5%)

**효과**:
- ✅ 1-A의 속도 + 1-B의 정밀도
- ✅ 서로의 약점 보완
- ✅ 45.4%는 Both로 검증 (신뢰도 ↑)

**단점**:
- ❌ 복잡도 증가
- ❌ LLM 비용 발생

**결론**: **Best performance, justified complexity**

---

## 7. 종합 분석 및 권장사항

### 중요도 순위

```
1위: Stage 1-A (Value/FTS)        - Core engine (75.7% 기여)
2위: Stage 1-B (Type + LLM)       - Essential complement (24.3% 독립 기여)
3위: Hybrid approach              - Optimal solution (+2.6%p boost)
```

### 각 Stage의 역할

**Stage 1-A**: **"Broad Net"** (넓게 잡기)
- 엔티티 이름으로 빠르게 많은 후보 수집
- High Recall, Low Precision
- 기본 엔진 역할

**Stage 1-B**: **"Precision Strike"** (정밀 타격)
- Type 정보로 의미적으로 필터링
- Low Recall, High Precision
- 보완 엔진 역할

**Hybrid**: **"Best of Both Worlds"**
- 1-A의 광범위 검색 + 1-B의 정밀 필터링
- 서로의 약점 보완
- 45.4% 중복 = confidence boost

---

## 8. 최적화 방향

### Stage 1-A 개선

1. **Precision 향상** (현재 3.62%)
   - Query expansion 개선
   - Metadata weighting 조정
   - 더 정확한 entity extraction

2. **FTS 설정 튜닝**
   - Match threshold 조정
   - Tokenization 개선

### Stage 1-B 개선

1. **LLM Filtering 완화** (현재 96.5% 제거)
   - Temperature 조정 (현재 0.1 → 0.2?)
   - Threshold 완화
   - Few-shot examples 추가

2. **비용 최적화**
   - Batch processing
   - Smaller model 시도 (gpt-4o-mini → gpt-3.5-turbo?)
   - Caching 강화

3. **Type 다양성 증가**
   - 현재 평균 2.69 types/entity
   - → 3-4 types로 확장하여 coverage 증가

### Hybrid 최적화

1. **'Both' source 활용**
   - 45.4%가 양쪽 모두 찾음
   - → Re-ranking 시 confidence score로 활용

2. **Adaptive routing**
   - Simple queries → Stage 1-A만
   - Complex queries → Full Hybrid
   - Cost vs Performance trade-off

---

## 9. ROI 분석

### Stage 1-B의 가치

**비용**:
- LLM API 비용 (21,864 evaluations)
- 추가 구현 복잡도
- 느린 속도

**이익**:
- +93 supporting facts (24.3%)
- +2.6%p Recall improvement
- +63 queries rescued (32.6%)
- 3.3배 higher Precision (3.62% → 12.09%)

**ROI**: **Positive** ✅
- Recall 개선이 비용을 정당화
- 특히 production에서 Precision 향상은 downstream 비용 절감

---

## 10. 결론 및 권장사항

### 핵심 결론

1. **Stage 1-A가 더 중요** (75.7% vs 69.7% 기여)
   - 하지만 **Stage 1-B 없이는 불완전**
   
2. **둘 다 필수적**
   - 각각 **32%의 쿼리**에서 독립적으로 필요
   
3. **Hybrid가 최선**
   - +2.6%p Recall boost
   - 45.4% Both = high confidence

### 시스템 선택 가이드

| 상황 | 권장 | 이유 |
|------|------|------|
| **Production (품질 중시)** | ✅ Hybrid | 최고 Recall (78.3%) |
| **Prototype (속도 중시)** | ✅ Stage 1-A only | 빠름, 75.7% Recall 충분 |
| **비용 제약** | ✅ Stage 1-A only | LLM 비용 없음 |
| **High precision 필요** | ✅ Hybrid | 1-B의 12.09% precision |
| **Research/Baseline** | ✅ Stage 1-A only | 간단, 재현 가능 |

### 최종 권장사항

**현재 시스템 유지 (Hybrid) ✅**

이유:
1. 최고 성능 (78.3% Recall)
2. Stage 1-B의 24.3% 독립 기여가 큼
3. ROI positive
4. Production-ready quality

**단, 다음 최적화 진행**:
1. LLM filtering 완화 (96.5% → 90%?)
2. 'Both' source를 confidence score로 활용
3. Cost optimization (batching, caching)

---

## 📈 추가 인사이트

### Query Type별 차이?

추후 분석 가능:
- Bridge vs Comparison questions
- Stage 1-A vs 1-B 선호도 차이?

### Supporting Facts 특성?

- 어떤 type의 facts를 각 stage가 잘 찾나?
- Title length, metadata richness와의 상관관계?

**향후 연구 방향** 🚀
