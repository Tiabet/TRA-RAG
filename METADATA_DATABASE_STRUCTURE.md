# 메타데이터 데이터베이스 구조 분석

## 📊 데이터베이스 개요

- **파일**: `HotpotQA/metadata_v2.db`
- **크기**: 5.21 MB
- **총 엔트리**: 1,993개
- **데이터베이스 엔진**: SQLite 3 with FTS5 (Full-Text Search)

## 🗄️ 테이블 스키마

### 1. `metadata` 테이블 (Main Table)

```sql
CREATE TABLE metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL UNIQUE,          -- 엔티티 이름 (중복 불가)
    title_normalized TEXT,                -- 검색용 정규화된 제목
    type TEXT,                            -- 메인 타입 (Person, WorkOfArt, Organization 등)
    subtype TEXT,                         -- 세부 타입 (Athlete, Film, Company 등)
    metadata_json TEXT NOT NULL,          -- JSON 형식의 전체 메타데이터
    searchable_text TEXT                  -- 모든 value를 flatten한 검색용 텍스트
)
```

**인덱스**:
- `idx_title_normalized`: 빠른 제목 검색
- `idx_type_subtype`: 타입 기반 필터링

### 2. `metadata_fts` 테이블 (Full-Text Search)

```sql
CREATE VIRTUAL TABLE metadata_fts USING fts5(
    title,
    searchable_text,
    content='metadata',
    content_rowid='id'
)
```

- FTS5 엔진을 사용한 전문 검색
- title과 searchable_text에서 빠른 검색 가능
- 테스트 검색 'university': 123개 매칭

## 📈 타입 분포

| Type | Count | Percentage |
|------|-------|------------|
| Person | 579 | 29.1% |
| WorkOfArt | 521 | 26.1% |
| Organization | 287 | 14.4% |
| Location | 208 | 10.4% |
| Event | 177 | 8.9% |
| Concept | 116 | 5.8% |
| Product | 55 | 2.8% |
| BiologicalEntity | 49 | 2.5% |
| OrganizationCluster | 1 | 0.1% |

## 📋 서브타입 분포 (Top 20)

| Type / Subtype | Count |
|----------------|-------|
| WorkOfArt / Film | 154 |
| Person / Athlete | 119 |
| Person / HistoricalFigure | 83 |
| Organization / Company | 80 |
| Event / SportsEvent | 79 |
| WorkOfArt / Album | 76 |
| Location / Building | 64 |
| WorkOfArt / Song | 64 |
| WorkOfArt / TelevisionSeries | 61 |
| Person / Musician | 59 |
| WorkOfArt / Book | 52 |
| Person / Politician | 51 |
| Person / Actor | 50 |
| Person / Writer | 47 |
| BiologicalEntity / Species | 35 |
| Event / HistoricalEvent | 34 |
| Organization / EducationalInstitution | 33 |
| Person / Artist | 27 |
| Event / CulturalEvent | 26 |
| Organization / GovernmentAgency | 26 |

## 🔍 메타데이터 JSON 구조

### 공통 필드 (모든 엔티티)

```json
{
  "title": "엔티티 이름",
  "type": "메인 타입",
  "subtype": "세부 타입",
  "attributes": { /* 엔티티별 속성 */ },
  "relations": [ /* 다른 엔티티와의 관계 */ ]
}
```

### 1. Person (예시: Athlete)

```json
{
  "title": "José Raúl Delgado",
  "type": "Person",
  "subtype": "Athlete",
  "attributes": {
    "full_name": "José Raúl Delgado Díez",
    "birth_date": "August 25, 1960",
    "nationality": "Cuban",
    "sport": "baseball",
    "achievements": {
      "medals": [
        {
          "type": "gold",
          "event": "1992 Summer Olympics",
          "count": 1
        }
      ]
    }
  },
  "relations": [
    {
      "relation": "uncle_of",
      "target": {
        "name": "Lourdes Gourriel",
        "type": "Person",
        "subtype": "Athlete"
      }
    }
  ]
}
```

**주요 속성**:
- `full_name`: 전체 이름
- `birth_date`, `birth_place`: 출생 정보
- `nationality`: 국적
- `sport`: 종목 (Athlete의 경우)
- `achievements`: 업적, 메달, 수상 등

### 2. Organization (예시: EducationalInstitution)

```json
{
  "title": "Syracuse University",
  "type": "Organization",
  "subtype": "EducationalInstitution",
  "attributes": {
    "full_name": "Syracuse University",
    "common_names": ["Syracuse", "'Cuse", "SU"],
    "type": "private research university",
    "location": {
      "city": "Syracuse",
      "state": "New York",
      "country": "United States"
    },
    "historical_roots": {
      "original_institution": {
        "name": "Genesee Wesleyan Seminary",
        "type": "Organization",
        "subtype": "EducationalInstitution",
        "founded": "1831"
      },
      "establishment_year": 1870
    }
  },
  "relations": [
    {
      "relation": "traced_to",
      "target": {
        "name": "Genesee Wesleyan Seminary",
        "type": "Organization",
        "subtype": "EducationalInstitution"
      }
    }
  ]
}
```

**주요 속성**:
- `full_name`, `common_names`: 이름 정보
- `type`: 조직 타입 (대학, 회사 등)
- `location`: 위치 정보 (중첩된 dict)
- `historical_roots`: 역사적 배경 (deeply nested)
- `identification`: 추가 식별 정보

### 3. WorkOfArt (예시: Film)

```json
{
  "title": "The Wolf of Wall Street",
  "type": "WorkOfArt",
  "subtype": "Film",
  "attributes": {
    "release_year": 2013,
    "nationality": "American",
    "genre": ["biographical", "black comedy", "crime"],
    "director": {
      "name": "Martin Scorsese",
      "type": "Person",
      "subtype": "Director"
    },
    "writer": {
      "name": "Terence Winter",
      "type": "Person",
      "subtype": "Writer"
    },
    "cast": [
      {
        "name": "Leonardo DiCaprio",
        "type": "Person",
        "subtype": "Actor",
        "role": "Jordan Belfort",
        "is_producer": true
      }
    ],
    "plot_summary": "긴 줄거리 설명...",
    "collaborations": {
      "director_actor": {
        "count": 5,
        "films": [ /* 이전 협업 작품들 */ ]
      }
    }
  },
  "relations": []
}
```

**주요 속성**:
- `release_year`: 개봉 연도
- `genre`: 장르 (배열)
- `director`, `writer`: 제작진 (nested objects)
- `cast`: 출연진 배열 (deeply nested)
- `plot_summary`: 줄거리
- `based_on`: 원작 정보 (있는 경우)
- `collaborations`: 협업 이력 (매우 중첩된 구조)

### 4. Location (예시: Airport)

```json
{
  "title": "Cap-Haïtien International Airport",
  "type": "Location",
  "subtype": "Airport",
  "attributes": {
    "name": "Cap-Haïtien International Airport",
    "also_known_as": "Aéroport International de Cap-Haïtien",
    "location": {
      "city": "Cap-Haïtien",
      "country": "Haiti"
    },
    "serves": {
      "city": "Cap-Haïtien",
      "region": "northern Haiti and the Atlantic coast"
    },
    "facilities": { /* 시설 정보 */ }
  },
  "relations": [
    {
      "relation": "serves",
      "target": {
        "name": "Cap-Haïtien",
        "type": "Location",
        "subtype": "City"
      }
    }
  ]
}
```

## 🔗 Relations 구조

Relations는 엔티티 간의 관계를 표현합니다:

```json
{
  "relation": "관계 타입",
  "target": {
    "name": "대상 엔티티 이름",
    "type": "대상 타입",
    "subtype": "대상 서브타입"
  }
}
```

**자주 사용되는 관계 타입**:
- `traced_to`: ~에서 유래
- `founded_by`: ~에 의해 설립
- `maintains_relationship_with`: ~와 관계 유지
- `written_by`: ~에 의해 작성
- `performed_by`: ~에 의해 공연
- `associated_with`: ~와 연관
- `uncle_of`, `granduncle_of`: 가족 관계
- `serves`: 서비스 제공 (공항, 교통 등)

## 🎯 중첩 구조의 특징

### 1. 깊은 중첩 (Deep Nesting)

메타데이터는 **매우 깊게 중첩**될 수 있습니다:

```json
{
  "attributes": {
    "historical_roots": {
      "original_institution": {
        "founder": {
          "location": {
            "city": "Lima",
            "state": "New York"
          }
        }
      }
    }
  }
}
```

### 2. 배열 내 객체

cast, achievements 등은 **배열 안에 복잡한 객체**를 포함:

```json
{
  "cast": [
    {
      "name": "Leonardo DiCaprio",
      "type": "Person",
      "subtype": "Actor",
      "role": "Jordan Belfort",
      "is_producer": true
    }
  ]
}
```

### 3. 혼합 타입

동일한 필드가 다양한 타입을 가질 수 있음:
- `genre`: 문자열 또는 배열
- `director`: 객체 또는 객체 배열

## 🔎 검색 메커니즘

### 1. Title 검색 (빠름)
```python
db.search_by_entity("Baltic Cup", search_title_only=True)
```
- `title_normalized` 필드에서 LIKE 검색
- 인덱스 활용으로 빠름

### 2. 전체 값 검색 (포괄적)
```python
db.search_by_entity("Estonia", search_title_only=False)
```
- `searchable_text` 필드에서 LIKE 검색
- 모든 중첩된 값들을 flatten하여 검색

### 3. FTS 검색 (빠르고 포괄적)
```python
db.search_by_entity_fts("Baltic Cup")
```
- FTS5 엔진 활용
- title과 searchable_text에서 전문 검색
- **matched_fields** 반환: 어떤 필드가 매칭되었는지 추적

### 4. 타입 필터링
```python
db.search_by_entity_fts(
    "Baltic Cup",
    entity_type="Event",
    entity_subtype="SportsEvent"
)
```
- 타입 필터와 함께 검색
- 결과가 없으면 자동으로 타입 필터 없이 재검색 (Fallback)

### 5. 타입별 검색
```python
db.search_by_type("Person", "Athlete")
```
- 특정 타입/서브타입의 모든 passage 검색
- Stage 1-B (Type Search)에서 사용

## 🛠️ 데이터베이스 유틸리티

### 초기화
```bash
python init_database.py --recreate
```

### 검사
```bash
python inspect_metadata_db.py
```

### 상세 뷰
```bash
# 제목으로 검색
python view_metadata.py --title "Syracuse University"

# 타입별 예시
python view_metadata.py --type Person --subtype Athlete
python view_metadata.py --type WorkOfArt --subtype Film
```

## 📊 성능 특성

- **데이터베이스 크기**: 5.21 MB (1,993 엔티티)
- **평균 엔티티 크기**: ~2.6 KB
- **FTS 인덱스**: 자동 관리, rebuild 시간 < 1초
- **검색 성능**:
  - Title 검색: < 10ms
  - FTS 검색: < 50ms
  - Type 필터링: < 20ms

## 🎨 메타데이터 품질

### 강점
1. **구조화된 타입 시스템**: Type + Subtype hierarchy
2. **풍부한 Relations**: 엔티티 간 명시적 관계
3. **중첩된 컨텍스트**: 깊은 속성 계층으로 풍부한 정보
4. **유연한 검색**: Title, FTS, Type 등 다양한 검색 방식

### 주의사항
1. **깊은 중첩**: 일부 메타데이터는 5-6단계 이상 중첩
2. **비일관성**: 같은 타입이라도 attributes 구조가 다를 수 있음
3. **텍스트 길이**: plot_summary 등 매우 긴 텍스트 포함
4. **Relations 포맷**: relation_type이 일부 누락된 경우 있음

## 📝 사용 예시

### Stage 1-A: Entity Search
```python
from metadata_db import MetadataDB

db = MetadataDB('HotpotQA/metadata_v2.db')

# FTS 검색으로 엔티티 찾기
results = db.search_by_entity_fts(
    "Leonardo DiCaprio",
    entity_type="Person",
    entity_subtype="Actor"
)

for result in results:
    print(f"Title: {result['title']}")
    print(f"Type: {result['type']} / {result['subtype']}")
    print(f"Matched fields: {result['matched_fields']}")
```

### Stage 1-B: Type Search
```python
# 특정 타입의 모든 passage 가져오기
results = db.search_by_type("WorkOfArt", "Film")

print(f"Found {len(results)} films")
for result in results:
    metadata = result['metadata']
    print(f"- {result['title']} ({metadata['attributes'].get('release_year', 'N/A')})")
```

### Context 추출
```python
# 메타데이터에서 관련 컨텍스트 추출
def extract_context(metadata):
    """메타데이터를 LLM이 이해할 수 있는 텍스트로 변환"""
    attrs = metadata.get('attributes', {})
    
    # 중요한 필드만 추출
    context = []
    
    if 'description' in attrs:
        context.append(attrs['description'])
    
    if 'birth_date' in attrs:
        context.append(f"Born: {attrs['birth_date']}")
    
    if 'location' in attrs:
        loc = attrs['location']
        if isinstance(loc, dict):
            context.append(f"Location: {loc.get('city', '')}, {loc.get('state', '')}")
    
    return ' | '.join(context)
```

---

## 🔄 업데이트 이력

- **2025-01-10**: 초기 문서 작성
  - 1,993개 엔티티 분석
  - 9개 메인 타입, 50+ 서브타입
  - 검색 메커니즘 3가지 (Title, Full, FTS)
  - 깊은 중첩 구조 확인
