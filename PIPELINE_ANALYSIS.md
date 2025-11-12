# Current Multi-hop QA Pipeline Analysis

## 📊 Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     run_pipeline_200.py                         │
│                  (Main Orchestration Layer)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   multihop_pipeline.py                          │
│              (Per-Question Pipeline Controller)                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Step 1: Query Decomposition                               │ │
│  │ Step 2: Sequential Answering (batch-aware parallel)      │ │
│  │ Step 3: Final Answer Synthesis                            │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                    │                     │
         ↓                    ↓                     ↓
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ query_           │ │ sequential_      │ │ sequential_      │
│ decomposition.py │ │ answering.py     │ │ answering.py     │
│                  │ │                  │ │                  │
│ • LLM 호출       │ │ • Entity 추출    │ │ • 최종 답변 생성 │
│ • Sub-Q 생성     │ │ • Hybrid Retrieval│ │ • LLM Synthesis  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
                              │
                              ↓
                    ┌──────────────────┐
                    │ hybrid_          │
                    │ retrieval.py     │
                    │                  │
                    │ • Stage 1-A      │
                    │ • Stage 1-B      │
                    │ • Merge Results  │
                    └──────────────────┘
                              │
                              ↓
                    ┌──────────────────┐
                    │ metadata_db.py   │
                    │ (V1 - 구버전)     │
                    │                  │
                    │ • FTS5 검색      │
                    │ • Type 필터링    │
                    └──────────────────┘
```

---

## 🔄 Detailed Pipeline Flow

### **Level 1: Main Orchestration** (`run_pipeline_200.py`)

**Purpose:** 200개 질문 병렬 처리 및 평가

```python
async def evaluate_multihop_200(questions, max_workers=30):
    1. Initialize LLM client & DB
    2. Create semaphore (limit concurrency)
    3. For each question (parallel):
       └─→ process_question_with_safety()
            └─→ process_single_question()  # in multihop_pipeline.py
    4. Collect results with progress tracking
    5. Save checkpoints every 10 questions
    6. Calculate metrics (EM, F1, timing)
    7. Save final results
```

**Key Features:**
- ✅ Parallel processing (30 workers default)
- ✅ Automatic checkpointing
- ✅ LLM logging (all interactions saved)
- ✅ Detailed metrics collection
- ✅ Thread-safety with deep copies

**Output Files:**
- `Results/multihop_pipeline_200_results.json` - Final results
- `Results/multihop_pipeline_200_checkpoint.json` - Checkpoints
- `Results/Logs/llm_log_YYYYMMDD_HHMMSS.txt` - All LLM calls

---

### **Level 2: Per-Question Pipeline** (`multihop_pipeline.py`)

**Purpose:** 단일 질문을 3단계로 처리

```python
async def process_single_question(client, db, question_data):
    
    # Step 1: Query Decomposition
    decomp_result = await decompose_query(client, question)
    # → Returns: QueryDecomposition object
    #   - question_type: "bridge" | "comparison"
    #   - subquestions: List[SubQuestion]
    
    # Step 2: Sequential Answering
    answering_result = await answer_subquestions_sequential(
        client, db, decomposition
    )
    # → Each sub-question:
    #   1. Substitute [SQ{N}_Answer] placeholders
    #   2. Extract entities
    #   3. Hybrid retrieval
    #   4. Generate answer
    
    # Step 3: Final Answer Synthesis
    final_answer = await synthesize_final_answer(client, decomposition)
    # → LLM combines all sub-answers
    
    return result
```

**Data Flow:**
```
Question
  ↓
QueryDecomposition {
  question_type: "bridge",
  subquestions: [
    SubQuestion {
      id: "SQ1",
      question: "...",
      depends_on: [],
      answer: None,
      retrieved_passages: [],
      retrieval_info: {}
    },
    SubQuestion {
      id: "SQ2",
      question: "... [SQ1_Answer] ...",
      depends_on: ["SQ1"],
      answer: None,
      ...
    }
  ]
}
  ↓
(Sequential Answering fills in answers and passages)
  ↓
Final Answer: "..."
```

---

### **Level 3: Query Decomposition** (`query_decomposition.py`)

**Purpose:** 복잡한 질문을 sub-question들로 분해

```python
async def decompose_query(client, question):
    # LLM Prompt:
    # - Detect question type (bridge/comparison)
    # - Break into logical sub-questions
    # - Mark dependencies with [SQ{N}_Answer] placeholders
    
    # Returns:
    {
        'success': True,
        'decomposition': QueryDecomposition(
            question_type="bridge",
            subquestions=[...]
        )
    }
```

**Example:**
```
Question: "The Bee Cliff overlooks a river that is how many miles long?"

Decomposition:
  SQ1: "What river does The Bee Cliff overlook?"
       → Answer: "Watauga River"
  
  SQ2: "How long is the [SQ1_Answer]?"
       → Answer: "78.5 mi long"
       → (substituted: "How long is the Watauga River?")
```

**Key Classes:**
- `SubQuestion`: 개별 sub-question 데이터
- `QueryDecomposition`: 전체 분해 결과
- `substitute_answers()`: Placeholder 치환
- `build_context_from_previous()`: 이전 답변들을 context로 변환

---

### **Level 4: Sequential Answering** (`sequential_answering.py`)

**Purpose:** Sub-question들을 순차적으로 답변 (배치 병렬 처리)

```python
async def answer_subquestions_sequential(client, db, decomposition):
    
    # Build batches (dependency-aware)
    batches = build_batches(decomposition.subquestions)
    # Batch 1: [SQ1, SQ3] (no dependencies)
    # Batch 2: [SQ2] (depends on SQ1)
    # Batch 3: [SQ4] (depends on SQ2)
    
    for batch in batches:
        # Parallel processing within batch
        tasks = [
            answer_single_subquestion(client, db, sq, decomposition)
            for sq in batch
        ]
        await asyncio.gather(*tasks)
    
    return result
```

**Per Sub-Question Processing:**
```python
async def answer_single_subquestion(client, db, subquestion, decomp):
    
    1. Substitute Placeholders
       "How long is [SQ1_Answer]?" 
       → "How long is Watauga River?"
    
    2. Build Context from Previous
       context = "From SQ1: The river is Watauga River."
    
    3. Extract Entities
       entities = await extract_entities_from_subquestion(
           client, substituted_question, context
       )
       # Returns: [
       #   {entity_name: "Watauga River", possible_types: [["Location", "River"]]}
       # ]
    
    4. Hybrid Retrieval
       passages = await hybrid_retrieve_for_subquestion(
           client, db, substituted_question, entities
       )
       # → Stage 1-A + Stage 1-B
    
    5. Generate Answer
       answer = await generate_answer_from_passages(
           client, substituted_question, passages
       )
    
    6. Update SubQuestion object
       subquestion.answer = answer
       subquestion.retrieved_passages = passages
       subquestion.retrieval_info = {...}
```

**Batching Logic:**
- ✅ Dependency-aware: respects `depends_on` field
- ✅ Parallel within batch: independent sub-questions run concurrently
- ✅ Sequential across batches: wait for all in batch before next

---

### **Level 5: Hybrid Retrieval** (`hybrid_retrieval.py`)

**Purpose:** 두 가지 방식으로 passage 검색 후 병합

```python
async def hybrid_retrieve_for_subquestion(client, db, query, entities):
    
    # Stage 1-A: Value-based Entity Matching
    stage1a_results = []
    for entity in entities:
        results = db.search_by_entity_fts(  # FTS5 fast search
            entity_name=entity['entity_name']
        )
        
        if apply_llm_filter_stage1a:
            # LLM filters titles for relevance
            filtered = await llm_filter_stage1a_titles(
                client, query, results
            )
            stage1a_results.extend(filtered)
        else:
            stage1a_results.extend(results)
    
    # Stage 1-B: Type-based Filtering + LLM Title Filtering
    stage1b_results = []
    for entity in entities:
        for type_option in entity['possible_types']:
            # Type filter: get all entities of this type
            candidates = db.search_by_type(
                entity_type=type_option[0],
                entity_subtype=type_option[1] if len(type_option) > 1 else None
            )
            
            # LLM filters titles for relevance to query
            filtered = await llm_filter_titles(
                client, query, entity, candidates
            )
            stage1b_results.extend(filtered)
    
    # Stage 2: Merge Results (deduplicate by title)
    merged_results = merge_by_title(stage1a_results, stage1b_results)
    
    return merged_results
```

**Two-Stage Strategy:**

**Stage 1-A: Value Matching** (Fast, High Recall)
```
Entity: "Watauga River"
  ↓ FTS5 search in metadata_db
Matches: [
  "Watauga River" (entity),
  "Bee Cliff (Tennessee)" (mentions river in description)
]
  ↓ (Optional) LLM filters for relevance
Final: Top relevant passages
```

**Stage 1-B: Type Matching** (Structured, High Precision)
```
Entity: "Watauga River"
Possible Types: [["Location", "River"]]
  ↓ Type filter
All Rivers: [
  "Watauga River",
  "Mississippi River",
  "Amazon River",
  ...
]
  ↓ LLM filters with title matching prompt
Final: "Watauga River" (high confidence)
```

**Merge Strategy:**
- Deduplicate by title
- Preserve metadata from both stages
- Return top K passages (configurable)

---

### **Level 6: Database Layer** (`metadata_db.py` - V1)

**Current DB Structure:**

```sql
Table: metadata
├── id (INTEGER PK)
├── title (TEXT)
├── title_normalized (TEXT)
├── type (TEXT)
├── subtype (TEXT)
├── metadata_json (TEXT)
└── searchable_text (TEXT)  ← V1: Flat concatenated values

Table: metadata_fts (FTS5 index)
└── searchable_text
```

**Key Methods:**

1. **`search_by_entity_fts(entity_name, type, subtype)`**
   - FTS5 full-text search on `searchable_text`
   - Optional type/subtype filtering
   - Returns matching metadata entries

2. **`search_by_type(entity_type, entity_subtype)`**
   - SQL WHERE filter on type/subtype
   - Used in Stage 1-B
   - Returns all entities of given type

3. **`search_by_entity(entity_name)`**
   - Basic LIKE search (fallback)
   - Slower but more flexible

---

## 🎯 Key Data Structures

### **QueryDecomposition**
```python
class QueryDecomposition:
    question_type: str              # "bridge" | "comparison"
    subquestions: List[SubQuestion]
    
    def to_dict() -> Dict
```

### **SubQuestion**
```python
class SubQuestion:
    id: str                    # "SQ1", "SQ2", ...
    question: str              # Sub-question text
    depends_on: List[str]      # ["SQ1"] or []
    answer: Optional[str]      # Filled during answering
    reasoning: str             # Why this sub-question
    retrieved_passages: List[Dict]
    retrieval_info: Dict       # Entity extraction & retrieval details
```

### **Entity (Extracted)**
```python
{
    "entity_name": "Watauga River",
    "possible_types": [
        ["Location", "River"],
        ["Location", "Waterway"]
    ],
    "role": "main_entity"
}
```

### **Passage (Retrieved)**
```python
{
    "title": "Watauga River",
    "metadata": {
        "title": "Watauga River",
        "type": "Location",
        "subtype": "River",
        "attributes": {...}
    }
}
```

---

## 📊 Current Database Usage

### **Database:** `HotpotQA/metadata_v2.db` (V1)

**Used in:**
- `hybrid_retrieval.py` → `MetadataDB('HotpotQA/metadata_v2.db')`
- `run_pipeline_200.py` → `MetadataDB('HotpotQA/metadata_v2.db')`

**Features:**
- ✅ FTS5 full-text search
- ✅ Type/subtype filtering
- ✅ Normalized title search
- ❌ No passage mapping
- ❌ Flat searchable_text (not path-based)

### **New Database:** `HotpotQA/metadata_v3.db` (V2)

**Features:**
- ✅ Path-based storage (`searchable_paths`)
- ✅ Passage mapping (`passages`, `metadata_passage_mapping`)
- ✅ Original passage content stored
- ✅ FTS5 on paths (structured search)

**Not yet integrated into pipeline!**

---

## 🔧 Configuration & Parameters

### **Main Settings** (`run_pipeline_200.py`)
```python
max_workers = 30                    # Parallel questions
use_fts = True                      # Use FTS5 for search
apply_llm_filter_stage1a = True     # Filter Stage 1-A results
```

### **Retrieval Settings** (implicit in code)
```python
# Stage 1-A
max_stage1a_results = 10  # Per entity

# Stage 1-B  
max_stage1b_results = 5   # Per entity-type combination

# Merge
max_final_passages = 10   # Top K after merging
```

### **LLM Models**
```python
model = "openai/gpt-4o-mini"
temperature = 0.1
max_tokens = 1024  # Entity extraction
max_tokens = 2048  # Answer generation
```

---

## 📈 Performance Characteristics

### **Timing Breakdown** (Average per question)
```
Total: ~15-20s
├─ Decomposition: 2-3s (10-15%)
├─ Answering: 10-15s (70-80%)
│  ├─ Entity Extraction: 1-2s per sub-Q
│  ├─ Retrieval: 1-3s per sub-Q
│  └─ Answer Generation: 2-4s per sub-Q
└─ Synthesis: 2-3s (10-15%)
```

### **Parallelism**
- **Across questions:** 30 concurrent (high speedup)
- **Within question:** Batch-aware (dependency respecting)
- **Within batch:** Full parallelism (independent sub-Qs)

### **Resource Usage**
- Database: Read-only, shared across workers
- LLM API: No rate limit (Alice API)
- Memory: Deep copies prevent cross-contamination

---

## 🚨 Current Issues & Limitations

### **1. Database Version Mismatch**
- ❌ Pipeline uses `metadata_v2.db` (V1 - flat storage)
- ✅ New `metadata_v3.db` (V2 - path-based + passages) exists
- ❌ Not integrated yet

### **2. No Passage Content in Retrieval**
- ❌ Only metadata is retrieved, no original passage text
- ❌ LLM generates answers from metadata alone
- ✅ V3 DB has passages but not used

### **3. Stage 1-A LLM Filtering Overhead**
- ⚠️ `apply_llm_filter_stage1a=True` adds latency
- ⚠️ May filter out correct passages
- ⚠️ No passage content to verify

### **4. Flat searchable_text (V1 DB)**
- ❌ Cannot search specific fields (e.g., "director-name")
- ❌ All values concatenated into one string
- ✅ V2 DB solves this with path-based storage

### **5. Entity Type Ambiguity**
- ⚠️ `possible_types` can be wrong from LLM
- ⚠️ Stage 1-B retrieves all entities of that type
- ⚠️ LLM title filtering may not be enough

---

## 🎯 Recommended Changes (Next Steps)

### **Priority 1: Switch to metadata_v3.db**
```python
# In hybrid_retrieval.py and run_pipeline_200.py
- db = MetadataDB('HotpotQA/metadata_v2.db')
+ from metadata_db_v2 import MetadataDBV2
+ db = MetadataDBV2('HotpotQA/metadata_v3.db')
```

### **Priority 2: Use Passage Content**
```python
# In answer generation
passages_with_content = db.get_metadata_with_passages(entity_name)
# Include passage.content in prompt, not just metadata
```

### **Priority 3: Leverage Path-based Search**
```python
# More structured queries
results = db.search_by_path_pattern("director-name-Martin Scorsese")
# Better than flat text search
```

### **Priority 4: Optimize Retrieval**
- Reduce/remove Stage 1-A LLM filtering (use path search instead)
- Increase reliance on FTS5 path search
- Use passage content for better LLM filtering

---

## 📂 File Structure Summary

```
ChunkRAG_v2/
├── run_pipeline_200.py              # Main orchestration
├── multihop_pipeline.py             # Per-question pipeline
├── query_decomposition.py           # Query → Sub-questions
├── sequential_answering.py          # Sub-Q answering logic
├── hybrid_retrieval.py              # Two-stage retrieval
├── metadata_db.py                   # V1 DB (CURRENT)
├── metadata_db_v2.py                # V2 DB (NEW - not used yet)
├── passage_mapper.py                # Passage mapping tool
├── init_database_v2.py              # V2 DB initialization
├── llm_logger.py                    # LLM call logging
├── llm_evaluation.py                # Results evaluation
└── Prompt/
    ├── entity_extraction_prompt.py
    ├── subquestion_entity_extraction_prompt.py
    ├── title_filtering_prompt.py
    ├── subquestion_answering_prompt.py
    └── ...
```

---

## 🔄 Data Flow Diagram

```
User Question
    ↓
[Decomposition] → Sub-Questions (SQ1, SQ2, ...)
    ↓
[For each SQ]
    ├─→ Substitute [SQ{N}_Answer]
    ├─→ Extract Entities
    ├─→ Hybrid Retrieval
    │   ├─→ Stage 1-A (Value matching)
    │   └─→ Stage 1-B (Type filtering)
    ├─→ Merge Results
    ├─→ Generate Answer
    └─→ Update SQ.answer
    ↓
[Synthesis] → Final Answer
    ↓
Output
```

---

## ✅ Summary

**Current State:**
- ✅ Full working pipeline (200 questions in ~10-15 min)
- ✅ High parallelism (30 workers)
- ✅ Comprehensive logging
- ✅ Two-stage hybrid retrieval
- ✅ Dependency-aware sub-question batching

**Ready to Upgrade:**
- 🔄 Switch to `metadata_v3.db` (path-based + passages)
- 🔄 Use passage content in answer generation
- 🔄 Optimize retrieval with structured path search
- 🔄 Remove unnecessary LLM filtering overhead

