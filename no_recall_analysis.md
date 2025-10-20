# NO RECALL 케이스 상세 분석 (7개)

## 📊 전체 패턴 요약

| Query | 핵심 문제 | 추출한 것 | 필요한 것 | 문제 유형 |
|-------|----------|----------|----------|----------|
| #52 | 작곡가 이름 누락 | "The Prelude...", "artist" | "Krzysztof Penderecki" | 답변 타입 추출 |
| #59 | 영화 제목 누락 | "wrestler" | "Final Score", "Dave Bautista" | 답변 타입 추출 |
| #61 | 음식/음료 이름 누락 | "alcohol content" | "Rippchen", "Apfelwein" | 답변 타입 추출 |
| #145 | 지명 누락 | "truck road" | "Backford Cross", "A41 road" | 답변 타입 추출 |
| #152 | 구체적 정보 누락 | "school" | "Michelle Williams", "Anton Chekhov", "Cherry Orchard" | 답변 타입 추출 + 복잡한 문장 |
| #193 | 작품/인물 누락 | "brothers" | "Paul Corbould", "Marvel", "Guardians of Galaxy" | 답변 타입 추출 |
| #196 | 구체적 건물/작품 누락 | "university" | "Hall of Languages", "Addams Family", "Gomez", "Morticia" | **문서 제목 불일치** |

---

## 🔍 케이스별 상세 분석

### **[1/7] Query #52 - 오페라 작곡가**

**Question**: "How many operas are among the artist who composed The Prelude for Clarinet in B-flat major best known works?"

**Answer**: four operas

**필요 문서**:
- ✗ Prelude for Clarinet (Penderecki)
- ✗ Krzysztof Penderecki

**추출 엔티티**:
- "The Prelude for Clarinet in B-flat major" → Stage1A: 0, Stage1B: 16→9
- "artist" → Stage1A: 27, Stage1B: 104→5 

**검색 결과**: Opera 관련 문서들 (The Midsummer Marriage, Libuše, Arlecchino...)

**문제점**:
1. ❌ "artist"라는 일반 명사 추출 (답변 타입)
2. ❌ "Krzysztof Penderecki" 작곡가 이름 누락
3. "The Prelude..." 작품명은 추출했으나 DB에 정확한 매칭 없음 (Title: "Prelude for Clarinet (Penderecki)")

**개선 방향**: 
- "who composed [작품명]" → 작품명 우선 추출
- 작품명에서 작곡가 유추 가능한 경우 작곡가도 추출 고려

---

### **[2/7] Query #59 - 레슬러 영화 출연**

**Question**: "Which semi-retired professional wrestler appeared in 'Final Score?'"

**Answer**: David Michael Bautista Jr.

**필요 문서**:
- ✗ Final Score (2017 film)
- ✗ Dave Bautista

**추출 엔티티**:
- "semi-retired professional wrestler" → Stage1A: 6, Stage1B: 119→4

**검색 결과**: 다른 레슬러들 (Tiger Jeet Singh, Barry Windham, Johnny Candido...)

**문제점**:
1. ❌ "wrestler" = 답변 타입 (직업)
2. ❌ **"Final Score"** 영화 제목을 추출하지 않음 (큰따옴표로 명확히 표시됨!)
3. Stage1A에서 "wrestler"로 다른 레슬러들만 검색

**개선 방향**:
- **큰따옴표 내 제목은 무조건 추출** (영화, 책, 노래 등)
- Few-shot 예시에 영화 제목 추출 패턴 추가

---

### **[3/7] Query #61 - 음료 알코올 도수**

**Question**: "What is the alcohol content of the drink normally consumed alongside Rippchen?"

**Answer**: 4.8%–7.0%

**필요 문서**:
- ✗ Frankfurter Rippchen
- ✗ Apfelwein

**추출 엔티티**:
- "alcohol content" → Stage1A: 4, Stage1B: 0→0

**검색 결과**: 일반 알코올 관련 문서 (Sugars in wine, Malt liquor, Breathalyzer...)

**문제점**:
1. ❌ "alcohol content" = 속성/답변 타입
2. ❌ **"Rippchen"** 음식 이름 누락
3. ❌ "Apfelwein" 음료 이름 누락 (문맥상 추론 필요)

**개선 방향**:
- "alongside [음식명]" → 음식명 우선 추출
- 고유 명사(외국어 포함) 인식 강화

---

### **[4/7] Query #145 - 도로 위치**

**Question**: "What major truck road is located in Backford Cross?"

**Answer**: The A41

**필요 문서**:
- ✗ Backford Cross
- ✗ A41 road

**추출 엔티티**:
- "truck road" → Stage1A: 1, Stage1B: 0→0

**검색 결과**: Portland and Southwestern Railroad Tunnel (완전히 무관)

**문제점**:
1. ❌ "truck road" = 답변 타입
2. ❌ **"Backford Cross"** 지명 누락 (고유명사!)
3. ❌ "A41" 도로 번호 누락

**개선 방향**:
- "located in [지명]" → 지명 우선 추출
- 고유명사 (특히 지명) 인식 강화

---

### **[5/7] Query #152 - 배우 학교**

**Question**: "What school was the actress who appeared with Michelle Williams in the last play by Russian playwright Anton Chekhov attending when she was signed on for a talent holding deal?"

**Answer**: Juilliard School

**필요 문서**:
- ✗ Jessica Chastain on screen and stage (x2)
- ✗ The Cherry Orchard

**추출 엔티티**:
- "school" → Stage1A: 26, Stage1B: 56→6

**검색 결과**: Technical schools (Erwin Technical Center, Manual Career & Technical Center...)

**문제점**:
1. ❌ "school" = 답변 타입
2. ❌ **"Michelle Williams"** 배우 이름 누락
3. ❌ **"Anton Chekhov"** 작가 이름 누락
4. ❌ **"The Cherry Orchard"** 작품명 누락 (last play = Cherry Orchard)
5. 문장이 복잡하여 핵심 엔티티 추출 실패

**개선 방향**:
- 복잡한 문장에서도 **고유명사 우선** (사람 이름, 작품명)
- "appeared with [배우]", "by [작가]" → 명시적 패턴

---

### **[6/7] Query #193 - 특수효과 감독 형제**

**Question**: "What are the names of the brothers of the special effects supervisor, known for his work on the films based on the Marvel Comics superhero team?"

**Answer**: Chris Corbould and Neil Corbould

**필요 문서**:
- ✗ Paul Corbould (x2)
- ✗ Guardians of the Galaxy (film) (x2)

**추출 엔티티**:
- "brothers" → Stage1A: 14, Stage1B: 19→0

**검색 결과**: Brothers 관련 무관 문서 (Blood film, Brewer Fieldhouse, The Allisons, Avsenik Brothers Ensemble...)

**문제점**:
1. ❌ "brothers" = 답변 타입
2. ❌ **"Marvel Comics"** 누락
3. ❌ **"superhero team"** 누락
4. ❌ "Guardians of the Galaxy" 영화 제목 누락 (문맥상 추론 필요)
5. "Paul Corbould" 이름 누락 (추론 필요)

**개선 방향**:
- **"Marvel", "superhero team"** 같은 명확한 힌트 우선 추출
- "based on [작품/개념]" → 작품명 추출

---

### **[7/7] Query #196 - 대학 건물 (특수 케이스!)**

**Question**: "At what university can the building that served as the fictional household that includes Gomez and Morticia be found?"

**Answer**: Syracuse University

**필요 문서**:
- ✗ Hall of Languages, Syracuse University (x2)
- ✗ The Addams Family

**추출 엔티티**:
- "university" → Stage1A: 14, Stage1B: 53→5

**검색 결과**: 
- ✓ Syracuse University (5번째) ← **찾았지만 No Recall!**
- Washington State University, Duke University, Tufts University...

**문제점**:
1. ❌ "university" = 답변 타입
2. ❌ **"Gomez", "Morticia"** 캐릭터 이름 누락
3. ❌ **"Addams Family"** 작품명 누락
4. ⚠️ **문서 제목 불일치**: 
   - 필요: "Hall of Languages, Syracuse University"
   - 검색됨: "Syracuse University"
   - → **Syracuse University 문서에는 Hall of Languages 정보가 없을 가능성!**

**개선 방향**:
- **"includes [캐릭터]"** → 캐릭터 이름 추출
- 캐릭터 → 작품명 유추 ("Gomez + Morticia" = Addams Family)
- **Supporting facts가 너무 구체적**: "Hall of Languages" 건물명까지 요구

---

## 📈 문제 유형 통계

| 문제 유형 | 개수 | 케이스 |
|----------|------|--------|
| **답변 타입 추출** (What/Which/Who 뒤 단어) | 7/7 | 전부! |
| **영화/작품 제목 누락** | 4 | #59, #152, #193, #196 |
| **고유명사 누락** (인물/지명/음식) | 5 | #52, #61, #145, #152, #193 |
| **문서 제목 불일치** | 1 | #196 |
| **복잡한 문장 구조** | 2 | #152, #193 |

---

## 🎯 핵심 개선 방향

### 1. **프롬프트 강화 (즉시 가능)**

추가할 Few-shot 예시:

```
Example: 영화 제목 추출
❌ "Which wrestler appeared in 'Final Score?'"
   → Extract: "wrestler" (WRONG - answer type)
✓ Correct: "Final Score" (movie title in quotes!)

Example: 인물명 우선
❌ "What school was the actress who appeared with Michelle Williams..."
   → Extract: "school" (WRONG - answer type)
✓ Correct: "Michelle Williams", "Anton Chekhov"

Example: 문맥 힌트 우선
❌ "...known for his work on the Marvel Comics superhero team"
   → Extract: "brothers" (WRONG - answer type)
✓ Correct: "Marvel Comics", "superhero team"
```

### 2. **Named Entity Recognition 강화**

- 큰따옴표/작은따옴표 안의 텍스트 = 제목/인용 → 무조건 추출
- 대문자로 시작하는 연속 단어 = 고유명사 가능성
- 외국어/특수 철자 (Rippchen, Apfelwein) 인식

### 3. **구문 패턴 인식**

- "appeared in [작품]" → 작품명 추출
- "by [작가]" → 작가명 추출
- "alongside [음식]" → 음식명 추출
- "located in [지명]" → 지명 추출
- "includes [캐릭터]" → 캐릭터명 추출

### 4. **답변 타입 필터링**

절대 추출하지 말아야 할 패턴:
- What/Which/Who 바로 뒤: school, university, artist, wrestler, brothers, road, content
- 예외: 고유명사와 결합된 경우 (Syracuse University - 하지만 이것도 답변!)

---

## 💡 시스템적 한계

### Query #196의 특수성
- **문제**: Supporting fact가 "Hall of Languages, Syracuse University"처럼 **너무 구체적**
- Syracuse University 문서를 찾았지만, 그 안에 Hall of Languages가 언급되어 있지 않으면 No Recall
- **해결 불가능한 이유**: 
  1. "Hall of Languages"를 추출하려면 "Gomez + Morticia" → "Addams Family" → "촬영 장소" 추론 필요
  2. 이는 외부 지식 필요 (프롬프트: "Extract ONLY from question")
  
→ **이 케이스는 현재 시스템으로 해결 어려움** (외부 지식 그래프 필요)

---

## 📊 개선 가능성 평가

| Query | 개선 가능성 | 난이도 |
|-------|-----------|--------|
| #52 | ⭐⭐⭐ 높음 | 보통 - 작품명만 더 추출하면 됨 |
| #59 | ⭐⭐⭐⭐⭐ 매우 높음 | 쉬움 - 큰따옴표 제목 추출 |
| #61 | ⭐⭐⭐⭐ 높음 | 보통 - "Rippchen" 고유명사 인식 |
| #145 | ⭐⭐⭐⭐ 높음 | 쉬움 - "Backford Cross" 고유명사 |
| #152 | ⭐⭐ 낮음 | 어려움 - 복잡한 문장, 다중 엔티티 |
| #193 | ⭐⭐⭐ 보통 | 보통 - "Marvel", "superhero" 추출 |
| #196 | ⭐ 매우 낮음 | 매우 어려움 - 외부 지식 필요 |

**예상 개선**: 7개 중 **4-5개 개선 가능** (#59, #61, #145, #52, 부분적으로 #193)

---

## 🚀 다음 액션

1. **즉시 적용**: 프롬프트에 3개 negative 예시 추가 (#59, #152, #193 패턴)
2. **중기 개선**: Named Entity Recognition 로직 추가 (큰따옴표, 고유명사)
3. **장기 개선**: 외부 지식 그래프 연동 (캐릭터→작품, 작품→장소 매핑)
