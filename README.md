# ChunkRAG v2 - Metadata-based RAG System

LLM 기반 메타데이터를 활용한 Multi-hop Question Answering 시스템

## 📁 프로젝트 구조

```
ChunkRAG_v2/
├── Prompt/                              # LLM 프롬프트 정의
│   ├── type_schema.py                   # 엔티티 타입 스키마
│   ├── metadata_construction_prompt.py  # 메타데이터 생성 프롬프트
│   └── entity_extraction_prompt.py      # 엔티티 추출 프롬프트
│
├── build_metadata.py                    # 메타데이터 생성 (비동기)
├── extract_entities_from_dataset.py     # 질문에서 엔티티 추출
│
├── 2WikiMultihopQA/                     # 2WikiMultihopQA 데이터셋
├── HotpotQA/                            # HotpotQA 데이터셋
├── MuSiQue/                             # MuSiQue 데이터셋
│
├── analyze/                             # 분석 및 테스트 스크립트
│   ├── analyze_entity_results.py        # 엔티티 추출 결과 분석
│   ├── test_entity_extraction.py        # 엔티티 추출 테스트
│   ├── analyze_question_types.py        # 질문 타입 분포 분석
│   ├── analyze_original_1000.py         # 1000개 데이터셋 분석
│   └── ...                              # 기타 분석 스크립트
│
└── README_METADATA_BUILD.md             # 메타데이터 빌드 가이드
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
