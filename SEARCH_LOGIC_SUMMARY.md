# Metadata DB V2 - Search Logic Summary

## 📊 현재 구현된 Search 메서드들

### 1️⃣ `search_by_entity()` - 기본 검색
**위치:** Line 374-453  
**타입:** LIKE 검색 (정규 검색)

```python
def search_by_entity(
    entity_name: str,
    entity_type: Optional[str] = None,
    entity_subtype: Optional[str] = None,
    search_title_only: bool = False
)
```

**동작 방식:**
- `search_title_only=True`: title만 검색 (빠름)
- `search_title_only=False`: searchable_paths 전체 검색 (느리지만 포괄적)
- Type/Subtype 필터링 가능
- 매칭 실패 시 type 필터 제거하고 재검색 (fallback)

**SQL:**
```sql
SELECT id, title, type, subtype, metadata_json
FROM metadata
WHERE searchable_paths LIKE '%entity_name%'
  AND type = ? AND subtype = ?
```

**장점:** 간단하고 직관적
**단점:** 느림 (LIKE는 인덱스를 못 씀)

---

### 2️⃣ `search_by_entity_fts()` - FTS5 검색 ⭐ (추천)
**위치:** Line 454-533  
**타입:** Full-Text Search (FTS5)

```python
def search_by_entity_fts(
    entity_name: str,
    entity_type: Optional[str] = None,
    entity_subtype: Optional[str] = None
)
```

**동작 방식:**
1. 특수문자 제거 및 정규화
2. 여러 단어면 phrase search 사용 (`"leonardo dicaprio"`)
3. FTS5 인덱스로 빠른 검색
4. **매칭된 경로를 추출하여 반환** (`matched_paths`)
5. Type 필터링 실패 시 재검색 (fallback)

**SQL:**
```sql
SELECT m.id, m.title, m.type, m.subtype, m.metadata_json, m.searchable_paths
FROM metadata_fts f
JOIN metadata m ON f.rowid = m.id
WHERE metadata_fts MATCH 'leonardo AND dicaprio'
  AND m.type = 'Person' AND m.subtype = 'Actor'
```

**반환값:**
```python
[{
    'title': 'The Wolf of Wall Street',
    'type': 'WorkOfArt',
    'subtype': 'Film',
    'metadata': {...},
    'matched_paths': [
        'cast-name-Leonardo DiCaprio-role-Jordan Belfort-is_producer-True-Person-Actor',
        'director-name-Martin Scorsese-Person-Director'
    ]
}]
```

**장점:** 
- ⚡ 매우 빠름 (FTS5 인덱스)
- 매칭된 경로 반환으로 왜 매칭됐는지 알 수 있음
- phrase search 지원

**단점:** 
- FTS5 쿼리 문법 제약 (특수문자 처리 필요)

---

### 3️⃣ `search_by_type()` - 타입별 검색
**위치:** Line 577-619  
**타입:** 정확한 타입 매칭

```python
def search_by_type(
    entity_type: str,
    entity_subtype: Optional[str] = None
)
```

**동작 방식:**
- Type과 Subtype으로 필터링
- 특정 타입의 모든 엔티티 조회

**SQL:**
```sql
SELECT title, type, subtype, metadata_json
FROM metadata
WHERE type = 'Person' AND subtype = 'Actor'
```

**사용 예시:**
```python
# 모든 배우 찾기
actors = db.search_by_type("Person", "Actor")

# 모든 영화 찾기
films = db.search_by_type("WorkOfArt", "Film")
```

**장점:** type/subtype 인덱스로 빠름
**용도:** Stage 1-B 타입 필터링, 통계 수집

---

### 4️⃣ `search_by_path_pattern()` - 경로 패턴 검색
**위치:** Line 620-661  
**타입:** 경로 구조 기반 검색

```python
def search_by_path_pattern(path_pattern: str)
```

**동작 방식:**
- 특정 경로 패턴을 포함하는 엔티티 검색
- 매칭된 경로만 추출하여 반환

**사용 예시:**
```python
# 감독 정보가 있는 모든 작품
results = db.search_by_path_pattern("director-name")

# cast role 정보가 있는 작품
results = db.search_by_path_pattern("cast-role")

# 협업 정보가 있는 작품
results = db.search_by_path_pattern("collaborations")
```

**SQL:**
```sql
SELECT id, title, type, subtype, metadata_json, searchable_paths
FROM metadata
WHERE searchable_paths LIKE '%director-name%'
```

**반환값:**
```python
[{
    'title': 'The Wolf of Wall Street',
    'type': 'WorkOfArt',
    'subtype': 'Film',
    'metadata': {...},
    'matched_paths': [
        'director-name-Martin Scorsese-Person-Director'
    ]
}]
```

**장점:** 구조화된 정보 검색에 유용
**용도:** 특정 관계/속성을 가진 엔티티 찾기

---

### 5️⃣ `get_by_title()` - 정확한 제목 검색
**위치:** Line 565-576  
**타입:** 정확한 매칭

```python
def get_by_title(title: str) -> Optional[Dict]
```

**동작 방식:**
- 정확한 title로 단일 엔티티 조회

**SQL:**
```sql
SELECT title, metadata_json
FROM metadata
WHERE title = 'The Wolf of Wall Street'
```

**장점:** 가장 빠름 (title은 UNIQUE)
**용도:** 정확한 제목으로 메타데이터 가져오기

---

## 🔧 보조 메서드들

### `_find_matched_paths()` - 매칭 경로 추출
**위치:** Line 534-564

```python
def _find_matched_paths(searchable_paths: str, search_terms: List[str])
```

- searchable_paths를 파싱하여 검색어를 포함하는 경로만 필터링
- 200자 이상이면 truncate
- `search_by_entity_fts()`에서 사용

### `build_searchable_paths()` - 경로 문자열 생성
**위치:** Line 309-318

```python
def build_searchable_paths(metadata: Dict) -> str
```

- `extract_paths()`로 경로 리스트 추출
- 쉼표로 연결하여 단일 문자열 생성
- FTS5 인덱싱을 위한 포맷

---

## 🎯 추천 사용 패턴

### Pattern 1: 엔티티 이름으로 검색 (가장 일반적)
```python
# FTS5 사용 (빠르고 추천)
results = db.search_by_entity_fts("Leonardo DiCaprio")

# 타입 필터링 추가
results = db.search_by_entity_fts(
    "Leonardo DiCaprio", 
    entity_type="Person", 
    entity_subtype="Actor"
)
```

### Pattern 2: 특정 타입의 모든 엔티티
```python
# 모든 영화
films = db.search_by_type("WorkOfArt", "Film")

# 모든 사람
people = db.search_by_type("Person")
```

### Pattern 3: 구조 기반 검색
```python
# 감독 정보가 있는 작품
films_with_director = db.search_by_path_pattern("director-name")

# 올림픽 메달 정보가 있는 선수
athletes_with_medals = db.search_by_path_pattern("olympic_participation-medal")
```

### Pattern 4: 정확한 제목
```python
# 단일 엔티티 조회
entity = db.get_by_title("The Wolf of Wall Street")
```

---

## 📈 성능 비교

| 메서드 | 속도 | 정확도 | 사용 시기 |
|--------|------|--------|-----------|
| `search_by_entity_fts()` | ⚡⚡⚡ 매우 빠름 | 높음 | 일반 검색 (추천) |
| `search_by_entity()` | 🐌 느림 | 높음 | FTS 사용 불가 시 |
| `search_by_type()` | ⚡⚡ 빠름 | 정확 | 타입 필터링 |
| `search_by_path_pattern()` | ⚡ 보통 | 정확 | 구조 검색 |
| `get_by_title()` | ⚡⚡⚡ 가장 빠름 | 정확 | 정확한 제목 |

---

## 🔍 FTS5 쿼리 문법

### 기본 검색
```sql
MATCH 'leonardo'                    -- 단일 단어
MATCH 'leonardo dicaprio'           -- OR 검색 (둘 중 하나)
MATCH 'leonardo AND dicaprio'       -- AND 검색 (둘 다 포함)
MATCH '"leonardo dicaprio"'         -- Phrase 검색 (정확한 순서)
```

### 현재 구현 로직
```python
# 단일 단어: 그대로
'leonardo' → MATCH 'leonardo'

# 여러 단어: phrase search
'leonardo dicaprio' → MATCH '"leonardo dicaprio"'
```

---

## 💡 개선 가능한 부분

1. **Fuzzy Search**: 오타 허용 검색 추가
2. **Ranking**: 관련도 순 정렬
3. **Highlighting**: 매칭된 부분 하이라이트
4. **Caching**: 자주 검색되는 쿼리 캐싱
5. **Batch Search**: 여러 엔티티 동시 검색

