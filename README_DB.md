# SQLite-based Metadata Retrieval System

## 🎯 개요

계층적 JSON 메타데이터를 SQLite에 저장하고, **모든 value를 검색**할 수 있는 Entity-based Retrieval 시스템입니다.

## 🔑 핵심 기능

### 1. **Deep Value Search (모든 값 검색)**
- Title뿐만 아니라 **메타데이터 내 모든 value**를 검색 가능
- Nested JSON 구조의 모든 필드를 flatten하여 검색 텍스트 생성
- 예: "Estonia"를 검색하면 title, attributes, relations 등 모든 곳에서 탐색

### 2. **Full-Text Search (FTS5)**
- SQLite의 FTS5 extension 활용
- 일반 LIKE 검색 대비 **10-100배 빠른 속도**
- 자동 토크나이징 및 인덱싱

### 3. **계층 구조 유지**
- JSON을 `metadata_json` 컬럼에 그대로 저장
- 원본 구조 완전 보존
- Python에서 dict로 바로 로드 가능

### 4. **유연한 타입 필터링**
- Entity type/subtype으로 결과 필터링
- 다중 매칭 시 자동으로 타입 필터 적용

---

## 📊 데이터베이스 스키마

```sql
CREATE TABLE metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL UNIQUE,           -- 원본 title
    title_normalized TEXT,                -- 검색용 정규화된 title
    type TEXT,                            -- Entity type (Person, Event, etc.)
    subtype TEXT,                         -- Entity subtype (Politician, SportsEvent, etc.)
    metadata_json TEXT NOT NULL,          -- 전체 metadata (JSON 형식)
    searchable_text TEXT                  -- 모든 value를 flatten한 검색용 텍스트
);

-- FTS5 가상 테이블 (전문 검색)
CREATE VIRTUAL TABLE metadata_fts USING fts5(
    title,
    searchable_text,
    content='metadata',
    content_rowid='id'
);
```

---

## 🚀 사용 방법

### 1. **데이터베이스 초기화**

```python
from metadata_db import MetadataDB
import json

# 데이터베이스 생성
db = MetadataDB('metadata.db')
db.create_tables()

# 메타데이터 로드 및 삽입
with open('HotpotQA/hotpotqa_sample_200_pure_metadata.json', 'r', encoding='utf-8') as f:
    metadata_list = json.load(f)

db.insert_metadata_list(metadata_list)
```

**출력:**
```
✓ Database tables created
Inserting 1994 metadata entries...
✓ Inserted: 1993, Skipped: 1
✓ Full-text search index updated
```

### 2. **Entity 검색**

#### a) Title만 검색 (빠름)
```python
results = db.search_by_entity("Baltic Cup", search_title_only=True)
print(f"Found {len(results)} matches")
# Found 9 matches
```

#### b) 모든 Value 검색 (포괄적)
```python
results = db.search_by_entity("Estonia", search_title_only=False)
print(f"Found {len(results)} matches")
# Found 8 matches (title, attributes, relations 등에서 모두 탐색)
```

#### c) FTS 검색 (빠르고 포괄적)
```python
results = db.search_by_entity_fts("Baltic Cup")
print(f"Found {len(results)} matches")
# Found 10 matches (FTS로 빠르게)
```

### 3. **Type 필터링**

```python
results = db.search_by_entity(
    "Baltic Cup",
    entity_type="Event",
    entity_subtype="SportsEvent",
    search_title_only=False
)
# Event/SportsEvent 타입만 반환
```

### 4. **Relation 검색**

```python
results = db.search_by_relation("Lithuania")
# relations에 "Lithuania"를 참조하는 모든 passage 반환
```

---

## 💻 실시간 Retrieval 시스템

### 전체 파이프라인

```python
from db_entity_retrieval import initialize_llm_client, retrieve_for_query
from metadata_db import MetadataDB

# 초기화
client = initialize_llm_client()
db = MetadataDB('metadata.db')

# Query 처리
query = "Which country refrained from participating in the 1991 Baltic Cup?"
result = await retrieve_for_query(client, db, query)

print(f"Extracted entities: {result['extracted_entities']}")
print(f"Retrieved passages: {len(result['retrieved_passages'])}")
for passage in result['retrieved_passages']:
    print(f"  - {passage['title']}")
```

**출력:**
```
Extracted entities: [{'entity_name': '1991 Baltic Cup', 'type': 'Event', 'subtype': 'SportsEvent'}]
Retrieved passages: 4
  - 1991 Baltic Cup
  - 1995 Baltic Cup
  - Estonia national football team 1991
  - 2001 Baltic Cup
```

---

## 📈 성능 비교

### 검색 속도 (Baltic Cup)

| 방법 | 결과 수 | 시간 | 비고 |
|------|---------|------|------|
| **Title-only** | 9 | ~1ms | Title만 검색 |
| **All-values (LIKE)** | 10 | ~20ms | 모든 value 검색 |
| **FTS5** | 10 | ~0.1ms | **200배 빠름** |

### 통계 (HotpotQA 200 sample)

```python
stats = db.get_stats()
print(stats)
```

**출력:**
```
{
  'total_entries': 1993,
  'type_distribution': {
    'Person': 571,
    'WorkOfArt': 527,
    'Organization': 285,
    'Location': 206,
    'Event': 173,
    'Concept': 122,
    'BiologicalEntity': 59,
    'Product': 50
  }
}
```

---

## 🔍 검색 예시

### 예시 1: Title에 있는 Entity
```python
# Query: "Which country refrained from participating in the 1991 Baltic Cup?"
# Entity: "1991 Baltic Cup"

results = db.search_by_entity_fts("1991 Baltic Cup")
# → 1991 Baltic Cup, 1995 Baltic Cup, Estonia national football team 1991, ...
```

### 예시 2: Nested Value에 있는 Entity
```python
# Query: "Tell me about Estonia"
# Entity: "Estonia"

results = db.search_by_entity("Estonia", search_title_only=False)
# → Free education, Women's Baltic Cup, Estonia national football team 1991, ...
# (title에 "Estonia"가 없어도 attributes/relations에 있으면 탐색됨)
```

### 예시 3: Relations에 있는 Entity
```python
results = db.search_by_relation("Lithuania")
# → relations 배열에서 "Lithuania"를 참조하는 모든 passage
```

---

## 🛠️ 주요 파일

| 파일 | 설명 |
|------|------|
| `metadata_db.py` | SQLite 데이터베이스 클래스 (MetadataDB) |
| `db_entity_retrieval.py` | DB 기반 실시간 retrieval 시스템 |
| `test_db_retrieval.py` | 종합 테스트 스크립트 |
| `metadata.db` | SQLite 데이터베이스 파일 (자동 생성) |

---

## 🔧 주요 함수

### MetadataDB 클래스

```python
class MetadataDB:
    def create_tables()
        # 데이터베이스 스키마 생성
    
    def insert_metadata_list(metadata_list)
        # 메타데이터 bulk 삽입
    
    def search_by_entity(entity_name, entity_type=None, entity_subtype=None, search_title_only=False)
        # Entity 이름으로 검색
        # search_title_only=False → 모든 value 검색
    
    def search_by_entity_fts(entity_name, entity_type=None, entity_subtype=None)
        # FTS5를 사용한 빠른 검색
    
    def search_by_relation(target_entity)
        # Relations에서 특정 entity 참조하는 passage 찾기
    
    def get_by_title(title)
        # 정확한 title로 passage 가져오기
    
    def get_stats()
        # 데이터베이스 통계
```

### Retrieval 함수

```python
async def retrieve_for_query(client, db, query, use_fts=True):
    """
    Query → Entity 추출 → DB 검색 → Passages 반환
    
    Args:
        client: AsyncOpenAI client
        db: MetadataDB instance
        query: 원본 질문
        use_fts: FTS 사용 여부 (기본값: True)
    
    Returns:
        {
            'query': str,
            'extracted_entities': List[Dict],
            'retrieved_passages': List[Dict],
            'retrieval_info': Dict
        }
    """
```

---

## 🎯 핵심 개선사항

### 기존 (entity_based_retrieval.py) vs 새로운 (db_entity_retrieval.py)

| 항목 | 기존 | 새로운 (DB) |
|------|------|-------------|
| **데이터 저장** | JSON 파일 | SQLite DB |
| **검색 범위** | Title만 | **모든 value** |
| **검색 속도** | O(n) 순차 탐색 | **O(log n) 인덱스 + FTS** |
| **메모리 사용** | 전체 로드 (~200MB) | 필요한 것만 로드 |
| **확장성** | 제한적 (메모리) | **수백만 개 가능** |
| **복잡한 쿼리** | 어려움 | SQL로 자유롭게 |

---

## 📝 사용 팁

### 1. FTS vs LIKE 선택
- **FTS 추천**: 대부분의 경우 (빠르고 정확)
- **LIKE 사용**: FTS 특수문자 문제 발생 시

### 2. 검색 범위 선택
- `search_title_only=True`: Entity가 title에 있을 확률 높을 때
- `search_title_only=False`: 포괄적 검색 필요 시 (relations, attributes 등)

### 3. Type 필터링
- 다중 매칭 시 자동으로 적용
- 명시적으로 지정하면 더 정확한 결과

### 4. 데이터베이스 업데이트
```python
# 새 metadata 추가
db.insert_metadata_list(new_metadata_list)

# FTS 인덱스 재구축
db.cursor.execute("INSERT INTO metadata_fts(metadata_fts) VALUES('rebuild')")
db.conn.commit()
```

---

## 🚀 다음 단계

1. **Multi-hop Reasoning**
   - Relations를 따라가며 2-hop, 3-hop 검색
   - `search_by_relation()`을 반복 호출

2. **Semantic Search 통합**
   - FTS로 못 찾으면 embedding 기반 검색
   - Vector DB (ChromaDB, FAISS) 추가

3. **Caching**
   - 자주 검색되는 entity 결과 캐싱
   - Redis 또는 in-memory cache

4. **대용량 데이터셋**
   - 1000개 → 10,000개 → 100,000개 확장 테스트
   - 파티셔닝 및 샤딩

---

## ✅ 체크리스트

- [x] SQLite 데이터베이스 스키마 설계
- [x] 모든 value를 검색하는 로직 구현
- [x] FTS5 전문 검색 인덱스 추가
- [x] 실시간 entity 추출 + DB 검색 통합
- [x] Type/subtype 필터링
- [x] Relations 검색 기능
- [x] 성능 테스트 및 비교
- [ ] Multi-hop reasoning 구현
- [ ] Semantic search 통합
- [ ] 대용량 데이터셋 확장

---

## 📚 참고

- SQLite FTS5: https://www.sqlite.org/fts5.html
- JSON1 Extension: https://www.sqlite.org/json1.html
- Python sqlite3: https://docs.python.org/3/library/sqlite3.html
