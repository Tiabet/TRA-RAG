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

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필수 패키지 설치
pip install openai python-dotenv tqdm

# 환경 변수 설정 (.env)
ALICE_OPENAI_KEY=your_api_key_here
ALICE_CHAT_URL=https://your-api-endpoint/v1
```

### 2. 메인 파이프라인 실행

```bash
# HotpotQA 200개 질문 처리
python run_pipeline_200.py

# 결과 파일:
# - Results/multihop_pipeline_200_results.json (최종 결과)
# - Results/multihop_pipeline_200_checkpoint.json (체크포인트)
# - Results/Logs/llm_log_*.txt (LLM 로그)
```

### 3. 결과 평가

```bash
# Token-based 평가 (EM, F1, Precision, Recall)
python judge_F1.py --pred Results/multihop_pipeline_200_results.json

# LLM-based 평가 (GPT-4o-mini)
python llm_evaluation.py --pred Results/multihop_pipeline_200_results.json
```

### 3. 사용 예시

```python
from hybrid_retrieval import initialize_llm_client, retrieve_for_query
from metadata_db import MetadataDB

# 초기화
client = initialize_llm_client()
db = MetadataDB('metadata_v2.db')

# 검색
query = "Who proposed free education plan in Argentina?"
result = await retrieve_for_query(client, db, query, use_fts=True)

# 결과
print(f"Retrieved: {len(result['retrieved_passages'])} passages")
for passage in result['retrieved_passages']:
    print(f"  - {passage['title']} [{passage.get('source', 'unknown')}]")
```

## 🏗️ 시스템 아키텍처

### Hybrid Retrieval Pipeline

```
Query
  ↓
┌─────────────────────────────────────────────────┐
│ Entity Extraction (gpt-4o-mini, temp=0.1)       │
│ - 엔티티 추출 + Role 분류                        │
│ - 각 엔티티당 2-3개 possible types              │
└─────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────┐
│ Stage 1-A: Value/FTS Matching                   │
│ - SQLite FTS5로 엔티티 이름 검색                 │
│ - 평균 16개 passages 검색                        │
│ - Precision: 3.62% | Recall 기여: 75.7%         │
└─────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────┐
│ Stage 1-B: Type Filtering + LLM Filter          │
│ 1. Type DB query (multiple types 시도)          │
│    → 평균 109개 candidates                       │
│ 2. LLM Title Filtering (semantic matching)      │
│    → 96.5% 제거, 평균 3.8개만 통과               │
│ - Precision: 12.09% | Recall 기여: 69.7%        │
└─────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────┐
│ Stage 2: Merge & Deduplicate                    │
│ - Source tagging (stage1a/stage1b/both)         │
│ - 45.4%가 Both = High Confidence                │
│ - 최종 평균 18개 passages                        │
└─────────────────────────────────────────────────┘
  ↓
Retrieved Passages
```

### Multiple Types System

**문제**: 단일 타입 추출 시 LLM-DB 스키마 불일치
```python
# ❌ 이전 (단일 타입)
{
  "entity_name": "education",
  "type": "Concept",
  "subtype": "AcademicField"  # DB에 없으면 0개 매칭!
}
```

**해결**: 엔티티당 2-3개 possible types
```python
# ✅ 현재 (다중 타입)
{
  "entity_name": "education",
  "possible_types": [
    {"type": "Concept", "subtype": "SocialSystem"},      # 19개 매칭
    {"type": "Concept", "subtype": "EducationalSystem"}, # 4개 매칭
    {"type": "Concept", "subtype": "AcademicField"}      # 0개 (괜찮음)
  ]
}
```

**효과**: 평균 2.69 types/entity → Type mismatch 문제 해결

## 📈 성능 분석

### Stage별 기여도

```
Supporting Facts 383개 중:

┌──────────────────────────────────────────┐
│ Stage 1-A만:    116 facts (30.3%)       │  ← Value로만 찾음
│ Stage 1-B만:     93 facts (24.3%)       │  ← Type으로만 찾음
│ Both:           174 facts (45.4%)       │  ← 양쪽 다 (고신뢰!)
├──────────────────────────────────────────┤
│ Stage 1-A 총:   290 facts (75.7%)       │  ← Core engine
│ Stage 1-B 총:   267 facts (69.7%)       │  ← Essential complement
│ Hybrid:         383 facts (78.3%)       │  ← Best performance
└──────────────────────────────────────────┘
```

### Question Type별 성능

| Type | Total | Full Recall | Partial | No Recall |
|------|-------|-------------|---------|-----------|
| **Comparison** | 42 | **97.6%** (41) | 2.4% (1) | 0% |
| **Bridge** | 158 | 50.0% (79) | 44.3% (70) | 5.7% (9) |

**인사이트**: Comparison 질문이 Bridge보다 훨씬 쉬움

### Stage별 특성

| 지표 | Stage 1-A (Value/FTS) | Stage 1-B (Type + LLM) |
|------|---------------------|---------------------|
| **속도** | ⭐⭐⭐⭐⭐ 매우 빠름 | ⭐⭐ 느림 |
| **Recall** | ⭐⭐⭐⭐⭐ 75.7% | ⭐⭐⭐⭐ 69.7% |
| **Precision** | ⭐ 3.62% | ⭐⭐⭐ 12.09% (3.3배!) |
| **독립 기여** | 30.3% | 24.3% |
| **비용** | ⭐⭐⭐⭐⭐ 무료 | ⭐⭐ LLM API |

**결론**: 
- Stage 1-A가 더 중요하지만 (75.7% vs 69.7%)
- Stage 1-B 없이는 불완전 (각각 32%의 쿼리에서 독립적으로 필요)
- **Hybrid가 최선** (78.3% Recall)

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
