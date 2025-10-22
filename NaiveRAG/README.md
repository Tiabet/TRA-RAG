# NaiveRAG - 범용 QA 시스템

## 개요

다양한 Multi-hop QA 데이터셋에 적용 가능한 NaiveRAG 구현입니다.

### 특징
- **Embedding Model**: text-embedding-3-small
- **Generation Model**: openai/gpt-4o-mini
- **Chunking Strategy**: 각 passage를 하나의 청크로 사용 (고정 토큰 크기 없음)
- **Evaluation**: Gold context 기반 retrieval 성능 평가
- **Parallel Processing**: 멀티스레드 병렬 처리로 빠른 평가

### 지원 데이터셋
- HotpotQA
- 2WikiMultihopQA
- MuSiQue
- 기타 유사한 구조의 QA 데이터셋

## 시스템 구조

```
NaiveRAG/
├── naive_rag.py              # 핵심 구현 (PassageChunk, NaiveRAG 클래스)
├── evaluate_naive_rag.py     # 전체 평가 스크립트 (병렬 처리)
├── show_results.py           # 결과 요약 출력
└── README.md                 # 이 파일
```

## 사용 방법

### 1. 환경 설정

필요한 패키지:
```bash
pip install openai python-dotenv tqdm numpy
```

`.env` 파일 설정:
```
ALICE_OPENAI_KEY=your_api_key
ALICE_EMBED_URL=your_embed_url
ALICE_CHAT_URL=your_chat_url
```

### 2. 단일 쿼리 테스트

```bash
# 데이터 경로를 인자로 전달
python naive_rag.py <data_path> [index_path]

# 예시: HotpotQA
python naive_rag.py ../HotpotQA/hotpot.jsonl

# 예시: 2WikiMultihopQA
python naive_rag.py ../2WikiMultihopQA/2wiki.jsonl
```

첫 실행 시:
1. 데이터를 로드합니다
2. 각 passage에 대한 임베딩을 생성합니다 (시간 소요)
3. 인덱스를 저장합니다 (`<basename>_index.json`)
4. 샘플 질문으로 테스트합니다

출력 예시:
```
Question: In what year was the university where Sergei Aleksandrovich Tokarev was a professor founded?
Gold Answer: 1755

Top-5 Retrieved Passages:
1. [0.8234] Moscow State University
2. [0.7891] Sergei Aleksandrovich Tokarev
3. [0.6543] Russian Academy of Sciences
...

Retrieval Metrics:
  Recall: 1.0000
  Precision: 0.4000
  F1: 0.5714
```

### 3. 전체 데이터셋 평가

```bash
# 기본 사용
python evaluate_naive_rag.py <data_path>

# 옵션 지정
python evaluate_naive_rag.py <data_path> [index_path] [results_path] [options]
```

**사용 예시:**

```bash
# HotpotQA 전체 평가 (기본 설정)
python evaluate_naive_rag.py ../HotpotQA/hotpot.jsonl

# Top-10으로 검색, 200개만 평가
python evaluate_naive_rag.py ../HotpotQA/hotpot.jsonl --top-k 10 --limit 200

# 병렬 워커 수 조정 (기본값: 8)
python evaluate_naive_rag.py ../HotpotQA/hotpot.jsonl --max-workers 16

# 2WikiMultihopQA 평가
python evaluate_naive_rag.py ../2WikiMultihopQA/2wiki.jsonl

# 모든 옵션 사용
python evaluate_naive_rag.py ../MuSiQue/musique.jsonl \
  --top-k 10 \
  --limit 500 \
  --max-workers 12
```

**명령행 옵션:**
- `--top-k N`: 검색할 passage 개수 (기본값: 5)
- `--limit N`: 평가할 질문 개수 제한 (기본값: 전체)
- `--max-workers N`: 병렬 워커 수 (기본값: 8)

**출력 예시:**

```
============================================================
Overall Retrieval Performance
============================================================
Total Questions:   1000
Average Recall:    0.6234
Average Precision: 0.5123
Average F1:        0.5621

============================================================
Performance by Question Type
============================================================

bridge:
  Count:      600
  Recall:     0.6400
  Precision:  0.5200
  F1:         0.5742

comparison:
  Count:      400
  Recall:     0.6000
  Precision:  0.5000
  F1:         0.5455

============================================================
Performance by Difficulty Level
============================================================

easy:
  Count:      300
  Recall:     0.7200
  Precision:  0.5800
  F1:         0.6431

medium:
  Count:      500
  Recall:     0.6100
  Precision:  0.4900
  F1:         0.5440

hard:
  Count:      200
  Recall:     0.5500
  Precision:  0.4500
  F1:         0.4950
```

결과는 `<basename>_naive_rag_results.json` 파일에 자동 저장됩니다.

### 4. 결과 요약 보기

```bash
python show_results.py <results_path>

# 예시
python show_results.py ../HotpotQA/hotpot_naive_rag_results.json
python show_results.py ../2WikiMultihopQA/2wiki_naive_rag_results.json
```

출력 내용:
- 전체 평균 성능
- Top 5 최고 성능 질문
- Top 5 최저 성능 질문
- 질문 타입별 분포

## 평가 메트릭

### Retrieval 메트릭

**Sentence-level Recall:**
- 정의: 검색된 passage 중 gold supporting facts가 포함된 비율
- 계산: `correct_facts / gold_facts`
- 특징: 같은 문서에서 여러 문장이 supporting fact인 경우, 문서를 찾으면 모든 문장을 찾은 것으로 간주

**Title-level Precision:**
- 정의: 검색된 passage 중 gold context에 포함된 title의 비율
- 계산: `correct_titles / retrieved_count`
- 특징: Passage 단위로 평가

**F1 Score:**
- Recall과 Precision의 조화 평균
- 계산: `2 * (recall * precision) / (recall + precision)`

### 데이터 형식 요구사항

JSONL 파일의 각 라인은 다음 필드를 포함해야 합니다:

```json
{
  "_id": "문서 ID",
  "question": "질문 텍스트",
  "answer": "정답",
  "supporting_facts": [
    ["Title 1", 0],
    ["Title 1", 2],
    ["Title 2", 1]
  ],
  "context": [
    ["Title 1", ["Sentence 0", "Sentence 1", "Sentence 2", ...]],
    ["Title 2", ["Sentence 0", "Sentence 1", ...]],
    ...
  ],
  "type": "bridge" or "comparison" (optional),
  "level": "easy", "medium", or "hard" (optional)
}
```

## 성능 최적화

### 병렬 처리
- 기본 8개 워커로 병렬 평가
- CPU 코어 수에 맞게 `--max-workers` 조정 가능
- 1000개 질문 기준: 순차 ~150초 → 병렬 ~19초 (약 8배 향상)

### 메모리 관리
- 임베딩 벡터는 numpy array로 메모리 효율적 저장
- 인덱스 파일 저장/로드로 재평가 시 임베딩 재생성 불필요
- Batch 처리로 API 호출 최소화

### API 최적화
- Batch size 100으로 임베딩 생성 (조정 가능)
- Rate limit 고려한 에러 핸들링
- 재시도 로직 포함

## 주요 클래스 및 API

### PassageChunk
```python
class PassageChunk:
    """각 passage를 표현하는 청크"""
    def __init__(self, title: str, content: List[str], doc_id: str, passage_idx: int)
    def get_text(self) -> str  # 임베딩용 텍스트 반환
```

### NaiveRAG
```python
class NaiveRAG:
    """범용 NaiveRAG 시스템"""
    
    def load_data(self, jsonl_path: str)
        # 데이터 로드 및 gold mapping 생성
    
    def create_embeddings(self, batch_size: int = 100)
        # 모든 청크에 대한 임베딩 생성
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[PassageChunk, float]]
        # 코사인 유사도 기반 검색
    
    def generate_answer(self, query: str, retrieved_chunks: List) -> str
        # LLM을 사용한 답변 생성 (선택적)
    
    def evaluate_retrieval(self, query_doc_id: str, retrieved_chunks: List) -> Dict
        # Recall, Precision, F1 계산
    
    def save_index(self, output_path: str)
        # 인덱스 저장
    
    def load_index(self, index_path: str)
        # 저장된 인덱스 로드
```

## 문제 해결

### 1. API 에러
```
Error: OpenAI API error
```
→ `.env` 파일의 API 키와 URL 확인
→ Rate limit 초과 시 `batch_size` 줄이기

### 2. 메모리 부족
```
MemoryError: Unable to allocate array
```
→ `--limit` 옵션으로 평가 개수 제한
→ Batch size 줄이기

### 3. 느린 속도
```
Evaluation taking too long
```
→ `--max-workers` 늘리기 (CPU 코어 수만큼)
→ 인덱스 저장 후 재사용

### 4. 인덱스 파일 손상
```
Error loading index
```
→ 인덱스 파일 삭제 후 재생성
→ 디스크 공간 확인

## 확장 가능성

### 새로운 데이터셋 추가
1. JSONL 형식 변환 (위의 데이터 형식 참고)
2. `supporting_facts`와 `context` 필드 필수
3. `type`, `level` 필드는 선택적 (없으면 'unknown')

### 커스터마이징
```python
from naive_rag import NaiveRAG

# 커스텀 설정으로 초기화
rag = NaiveRAG()
rag.load_data("custom_data.jsonl")

# 더 큰 배치 사이즈
rag.create_embeddings(batch_size=200)

# 커스텀 평가
custom_results = rag.retrieve("My question", top_k=10)
```

## 성능 개선 아이디어

1. **Hybrid Retrieval**: BM25 + Dense retrieval 결합
2. **Reranking**: Cross-encoder 모델로 재정렬
3. **Query Expansion**: 쿼리 변형/확장으로 recall 향상
4. **Fine-tuning**: 도메인 특화 임베딩 모델
5. **Sentence-level Chunking**: Passage 대신 문장 단위 청크

## 라이센스

이 프로젝트는 ChunkRAG_v2의 일부입니다.
