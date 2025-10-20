# ChunkRAG v2 - Hybrid Metadata Retrieval System

**LLM 기반 메타데이터를 활용한 Multi-hop Question Answering 시스템**

[![Performance](https://img.shields.io/badge/Recall-78.3%25-brightgreen)]()
[![Success Rate](https://img.shields.io/badge/Success-100%25-blue)]()
[![Questions](https://img.shields.io/badge/Tested-200_queries-orange)]()

## � 핵심 성능

| 지표 | 결과 |
|------|------|
| **Overall Recall** | **78.3%** (383/489 supporting facts) |
| **검색 성공률** | **100%** (200/200 queries) |
| **No Recall** | **3.5%** (7/200 queries) |
| **Full Recall** | 59.0% (118/200 queries) |
| **처리 속도** | 0.40 queries/sec |

## �📁 프로젝트 구조

```
ChunkRAG_v2/
├── 📂 Core System
│   ├── hybrid_retrieval.py              # ⭐ Hybrid retrieval (Stage 1-A + 1-B + 2)
│   ├── metadata_db.py                   # SQLite FTS5 DB 관리
│   └── metadata_v2.db                   # 메타데이터 DB (5.3 MB)
│
├── 📂 Prompts
│   ├── Prompt/
│   │   ├── type_schema.py               # 엔티티 타입 스키마 (9 types, 50+ subtypes)
│   │   ├── entity_extraction_prompt.py  # ⭐ 엔티티 추출 (multiple types)
│   │   ├── metadata_construction_prompt.py  # 메타데이터 생성
│   │   └── llm_filtering_prompt.py      # LLM semantic filtering
│
├── 📂 Testing & Analysis
│   ├── test_hybrid_200.py               # ⭐ 최종 테스트 스크립트 (200 queries)
│   ├── analyze_hybrid_200.py            # 결과 분석 스크립트
│   ├── test_hybrid_200_results.json     # 상세 결과 (946 KB)
│   ├── test_hybrid_200_summary.txt      # 요약 통계
│   └── tests_archive/                   # 개발 과정 아카이브
│
├── 📂 Documentation
│   ├── PROJECT_SUMMARY.md               # ⭐ 프로젝트 종합 요약
│   ├── stage_comparison_analysis.md     # Stage 1-A vs 1-B 비교 분석
│   ├── no_recall_analysis.md            # No Recall 케이스 분석
│   └── README.md                        # 이 문서
│
└── 📂 Datasets
    ├── HotpotQA/                        # HotpotQA 200 샘플
    ├── 2WikiMultihopQA/                 # 2WikiMultihopQA 샘플
    └── MuSiQue/                         # MuSiQue 샘플
```

## 🚀 주요 기능

### 1. 메타데이터 생성 (`build_metadata.py`)
passage에서 구조화된 메타데이터를 LLM으로 추출합니다.

```bash
# 200개 샘플 처리
python build_metadata.py -i HotpotQA/hotpotqa_sample_200.json --concurrency 20

# 1000개 전체 데이터셋 처리
python build_metadata.py -i HotpotQA/hotpot.jsonl --concurrency 20
```

**추출되는 메타데이터:**
- Main Entities (이름, 타입, 서브타입, 관계)
- Hierarchical Events (이벤트, 날짜, 참여자)
- Attributes (숫자 정보, 시간 정보, 통계)
- Relations (엔티티 간 관계 그래프)

### 2. 엔티티 추출 (`extract_entities_from_dataset.py`)
질문에서 시작 엔티티를 추출합니다.

```bash
# 질문에서 엔티티 추출
python extract_entities_from_dataset.py -i HotpotQA/hotpotqa_sample_200.json --concurrency 20
```

**지원 기능:**
- 단일 엔티티 추출 (bridge 질문)
- 다중 엔티티 추출 (comparison 질문)
- 엔티티 타입 자동 분류

## 📊 성능 결과

### HotpotQA 200 샘플 테스트

**메타데이터 생성:**
- 성공률: 100% (1,994/1,994 passages)
- 처리 시간: 18:33 (1.79 it/s)
- 동시 처리: 20개

**엔티티 추출:**
- 성공률: 100% (200/200 questions)
- 처리 시간: 39초 (5.05 it/s)
- Bridge 질문: 평균 1.27개 엔티티
- Comparison 질문: 평균 1.98개 엔티티

## 🛠️ 환경 설정

### 필수 패키지
```bash
pip install openai python-dotenv tqdm
```

### 환경 변수 (.env)
```
ALICE_OPENAI_KEY=your_api_key_here
ALICE_CHAT_URL=https://your-api-endpoint/v1
```

## 📈 데이터셋 정보

| Dataset | Questions | Passages | Question Types |
|---------|-----------|----------|----------------|
| HotpotQA | 1,000 | 9,981 | Bridge (79%), Comparison (21%) |
| 2WikiMultihopQA | 1,000 | 10,000 | Compositional (46%), Comparison (25%), Bridge_comparison (19%), Inference (11%) |
| MuSiQue | 1,000 | 19,990 | 2-hop (50%), 3-hop (34%), 4-hop (17%) |
| **Total** | **3,000** | **39,971** | Multi-hop QA |

## 💡 사용 예시

### 메타데이터 생성 + 엔티티 추출 파이프라인

```bash
# 1. 메타데이터 생성
python build_metadata.py -i HotpotQA/hotpot.jsonl --concurrency 20

# 2. 엔티티 추출
python extract_entities_from_dataset.py -i HotpotQA/hotpot.jsonl --concurrency 20

# 결과 파일:
# - HotpotQA/hotpot_metadata.json     (passage 메타데이터)
# - HotpotQA/hotpot_entities.json     (질문 엔티티)
```

## 🔍 Hybrid Retrieval System

### 다중 타입 추출 (Multiple Types per Entity)

**문제**: 단일 타입 추출 시 LLM과 DB 스키마 불일치로 매칭 실패
- 예: LLM이 "Concept/AcademicField" 추출 → DB에는 "Concept/SocialSystem"만 존재

**해결**: 엔티티당 2-3개의 `possible_types` 추출

```python
# 이전 (단일 타입)
{
  "entity_name": "education system",
  "type": "Concept",
  "subtype": "AcademicField"  # DB에 없으면 매칭 실패!
}

# 현재 (다중 타입)
{
  "entity_name": "education system",
  "possible_types": [
    {"type": "Concept", "subtype": "EducationalSystem"},  # 4개 매칭
    {"type": "Concept", "subtype": "SocialSystem"},       # 19개 매칭
    {"type": "Concept", "subtype": "AcademicField"}       # 0개 (괜찮음)
  ]
}
```

### Hybrid Retrieval Pipeline

```
Query → Entity 추출
         ↓
┌─────────────────────────────────────┐
│ Stage 1-A: Value Matching          │
│ - Entity name FTS 검색              │
│ - 모든 metadata 값에서 검색          │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Stage 1-B: Type Filtering           │
│ 1. 여러 possible_types로 DB 검색     │
│    (Type 1 + Type 2 + Type 3...)   │
│ 2. LLM semantic filtering          │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Stage 2: Merge                      │
│ - Value + Type 결과 병합             │
│ - Title 기준 중복 제거               │
└─────────────────────────────────────┘
```

### 성능 개선

| 메트릭 | 이전 (단일 타입) | 현재 (다중 타입) | 개선율 |
|--------|-----------------|-----------------|--------|
| Types per entity | 1개 | 2-3개 | 200-300% |
| Type matching (Argentina education) | 0개 | 23개 | ∞ |
| 정답 발견 | 실패 | 성공 (#1위) | ✅ |

### 테스트

```bash
# Hybrid retrieval 종합 테스트 (3가지 쿼리)
python test_hybrid_retrieval.py
```

## 🔍 엔티티 타입 스키마

9개 주요 타입, 50+ 서브타입 지원:
- **Person**: Politician, Athlete, Artist, Writer, Scientist, etc.
- **Location**: Country, City, Landmark, NaturalPlace, Facility
- **Organization**: Company, GovernmentAgency, EducationalInstitution, etc.
- **WorkOfArt**: Book, Film, Song, Album, TelevisionSeries, etc.
- **Event**: HistoricalEvent, SportsEvent, CulturalEvent, etc.
- **Product**: Software, Hardware, Vehicle, etc.
- **BiologicalEntity**: Animal, Plant, Disease, Gene, etc.
- **Concept**: Scientific, Philosophical, Economic, Legal, etc.
- **OrganizationCluster**: MusicGroup, SportsTeam, PoliticalParty, etc.

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.

## 👥 기여

분석 스크립트 및 테스트 코드는 `analyze/` 폴더에서 확인할 수 있습니다.
