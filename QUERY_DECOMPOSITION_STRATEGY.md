# Query Decomposition Strategy for Multi-Dataset Support

## 📊 Dataset Analysis Summary

### **1. HotpotQA** (Current Support ✅)

**Question Types:**
- **Bridge** (79%): "A overlooks B. How long is B?"
- **Comparison** (21%): "Which was filmed first, A or B?"

**Characteristics:**
- 2-hop questions
- Simple dependency chains
- Clear bridge/comparison patterns

**Current Strategy:**
```python
# query_decomposition.py
- Detects: "bridge" | "comparison"
- Generates: 2 sub-questions typically
- Dependencies: SQ2 depends on SQ1
```

---

### **2. 2WikiMultihopQA** (New Dataset 🆕)

**Question Types:**
- **Compositional** (45.5%): "Where was the director of film X born?"
- **Comparison** (25.0%): "Who was born later, A or B?"
- **Bridge-Comparison** (19.0%): "Which film has the director who died earlier?"
- **Inference** (10.5%): "Who is the father-in-law of X?"

**Characteristics:**
- More diverse patterns
- Compositional = Bridge + attribute extraction
- Bridge-Comparison = hybrid type (bridge THEN compare)
- Inference = requires relationship reasoning

**Example Patterns:**

**Compositional:**
```
Q: "Where was the director of film Doctor Krishna born?"
SQ1: "Who is the director of film Doctor Krishna?"
     → Answer: "Phani Ramachandra"
SQ2: "Where was [SQ1_Answer] born?"
     → Answer: "Karnataka"
```

**Bridge-Comparison:**
```
Q: "Which film has the director who died earlier, A Doctor's Diary or Wild Rovers?"
SQ1: "Who is the director of A Doctor's Diary?"
SQ2: "Who is the director of Wild Rovers?"
SQ3: "When did [SQ1_Answer] die?"
SQ4: "When did [SQ2_Answer] die?"
SQ5: "Which is earlier: [SQ3_Answer] or [SQ4_Answer]?"
```

**Inference:**
```
Q: "Who is the father-in-law of Elizabeth Somerset?"
SQ1: "Who is the husband of Elizabeth Somerset?"
     → Answer: "William Herbert"
SQ2: "Who is the father of [SQ1_Answer]?"
     → Answer: "Henry Beaufort"
```

---

### **3. MuSiQue** (New Dataset 🆕)

**Question Types:**
- **Unknown/Mixed** (100%): No explicit type labels
- **Has gold decomposition** ✅

**Characteristics:**
- 2-4 hop questions (avg: 2.67)
- Complex reasoning chains
- Uses placeholder syntax: `#1`, `#2`, `#3`
- Highly compositional

**Decomposition Length Distribution:**
- 2 steps: 49.5%
- 3 steps: 34.0%
- 4 steps: 16.5%

**Example Patterns:**

**2-hop (property chaining):**
```
Q: "What body of water is by the headquarters location of Wipac?"
Step 1: "Wipac >> headquarters location"
        → "Buckingham"
Step 2: "Which is the body of water by #1?"
        → "River Great Ouse"
```

**3-hop (nested properties):**
```
Q: "What is the symbol of the Saints from the city where the headquarters 
    of the manufacturer of McAfee's Benchmark called?"
Step 1: "McAfee's Benchmark >> manufacturer"
        → "Sazerac Company"
Step 2: "#1 >> headquarters location"
        → "New Orleans"
Step 3: "what is the #2 saints symbol called"
        → "fleur-de-lis"
```

**4-hop (complex multi-entity):**
```
Q: "What major conflict is the country between the country that hosted 
    the tournament and the country where That Dam is from known for?"
Step 1: "Who hosted the tournament?"
        → "Thailand"
Step 2: "That Dam >> country"
        → "Laos"
Step 3: "what natural boundary lies between #1 and #2"
        → "Myanmar"
Step 4: "What major conflict is #3 known for?"
        → "one of the world's longest-running ongoing civil wars"
```

---

## 🎯 Unified Query Decomposition Strategy

### **Approach 1: Type-Agnostic LLM Decomposition** (Recommended)

**Concept:** Let LLM detect patterns and decompose naturally without rigid type constraints.

```python
UNIFIED_DECOMPOSITION_PROMPT = """
You are an expert at breaking down complex multi-hop questions into simpler sub-questions.

Question: {question}

Your task:
1. Analyze the reasoning chain needed to answer this question
2. Break it into logical sub-questions
3. Use placeholder syntax [SQ{N}_Answer] for dependencies
4. Each sub-question should be answerable from a single knowledge source

Guidelines:
- Identify what information is needed at each step
- Mark dependencies clearly (which sub-question depends on which)
- Keep sub-questions simple and focused
- Typical patterns:
  * Property lookup: "What is X's property Y?"
  * Comparison: "Which is greater/earlier/longer, A or B?"
  * Multi-entity: "Given A and B, what is the relationship?"
  * Inference: "Based on relationship R, who is X?"

Output format:
{
  "question_type": "compositional" | "comparison" | "inference" | "bridge" | "mixed",
  "reasoning": "Brief explanation of the decomposition strategy",
  "subquestions": [
    {
      "id": "SQ1",
      "question": "...",
      "depends_on": [],
      "reasoning": "Why this sub-question is needed"
    },
    ...
  ]
}
"""
```

**Advantages:**
- ✅ Works across all datasets
- ✅ Flexible to new patterns
- ✅ Leverages LLM's reasoning ability
- ✅ No need to pre-classify question types

**Disadvantages:**
- ⚠️ May generate inconsistent decompositions
- ⚠️ Requires validation

---

### **Approach 2: Pattern-Based Type Detection** (More Structured)

**Concept:** Detect question patterns, then apply type-specific decomposition.

```python
class QuestionPattern:
    # Bridge (A → B)
    BRIDGE = r"(?i).*(where|who|when|what).*(?:of|in|from|by).*"
    
    # Comparison (A vs B)
    COMPARISON = r"(?i).*(which|who|what).*(earlier|later|longer|more|less|first|last).*"
    
    # Compositional (property chains)
    COMPOSITIONAL = r"(?i).*(of|in).*(?:of|in).*"  # Multiple "of" patterns
    
    # Inference (relationships)
    INFERENCE = r"(?i).*(father|mother|spouse|child|sibling|relative).*"
    
    # Bridge-Comparison (hybrid)
    BRIDGE_COMPARISON = r"(?i).*(which).*(?:has|with).*(?:earlier|later|more|less).*"

def detect_question_type(question: str) -> str:
    """Detect question type from patterns"""
    patterns = {
        'bridge_comparison': QuestionPattern.BRIDGE_COMPARISON,
        'inference': QuestionPattern.INFERENCE,
        'comparison': QuestionPattern.COMPARISON,
        'compositional': QuestionPattern.COMPOSITIONAL,
        'bridge': QuestionPattern.BRIDGE
    }
    
    for qtype, pattern in patterns.items():
        if re.search(pattern, question):
            return qtype
    
    return 'unknown'

def decompose_by_type(question: str, qtype: str) -> QueryDecomposition:
    """Apply type-specific decomposition strategy"""
    
    if qtype == 'compositional':
        return decompose_compositional(question)
    elif qtype == 'bridge_comparison':
        return decompose_bridge_comparison(question)
    elif qtype == 'inference':
        return decompose_inference(question)
    elif qtype == 'comparison':
        return decompose_comparison(question)
    elif qtype == 'bridge':
        return decompose_bridge(question)
    else:
        return decompose_generic(question)  # Fallback to LLM
```

**Advantages:**
- ✅ More predictable decompositions
- ✅ Type-specific optimizations
- ✅ Easier to debug

**Disadvantages:**
- ⚠️ Requires maintaining patterns
- ⚠️ May miss edge cases

---

### **Approach 3: Hybrid (Best of Both)** ⭐ **RECOMMENDED**

**Concept:** Combine pattern detection with LLM flexibility.

```python
async def decompose_query_hybrid(
    client: AsyncOpenAI,
    question: str,
    use_type_detection: bool = True
) -> Dict:
    """
    Hybrid decomposition strategy:
    1. Attempt pattern-based type detection
    2. Use LLM for flexible decomposition
    3. Validate and refine
    """
    
    # Step 1: Pattern-based hint (optional)
    detected_type = None
    if use_type_detection:
        detected_type = detect_question_type(question)
    
    # Step 2: LLM decomposition with type hint
    prompt = UNIFIED_DECOMPOSITION_PROMPT.format(
        question=question
    )
    
    if detected_type and detected_type != 'unknown':
        prompt += f"\n\nHint: This question appears to be a '{detected_type}' type."
    
    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    decomposition = json.loads(response.choices[0].message.content)
    
    # Step 3: Validation
    validated = validate_decomposition(decomposition)
    
    return {
        'success': True,
        'detected_type': detected_type,
        'decomposition': validated
    }

def validate_decomposition(decomposition: Dict) -> QueryDecomposition:
    """
    Validate and refine decomposition:
    - Check dependency cycles
    - Ensure proper placeholder syntax
    - Verify all dependencies exist
    """
    # Validation logic...
    return QueryDecomposition(...)
```

**Advantages:**
- ✅ Pattern detection provides hints
- ✅ LLM handles complex cases
- ✅ Validation ensures correctness
- ✅ Best of both worlds

---

## 🔧 Implementation Plan

### **Phase 1: Extend Current Decomposition**

**File:** `query_decomposition.py`

**Changes:**
1. Update prompt to handle more question types
2. Remove hardcoded "bridge" | "comparison" constraint
3. Add support for 3-4 hop questions
4. Support `#1`, `#2` placeholder syntax (in addition to `[SQ{N}_Answer]`)

```python
# Current
question_type: Literal["bridge", "comparison"]

# New
question_type: str  # Any type: "compositional", "inference", "bridge_comparison", etc.
```

### **Phase 2: Add Pattern Detection** (Optional)

**New file:** `question_patterns.py`

```python
class QuestionTypeDetector:
    def detect(self, question: str) -> str:
        """Detect question type from patterns"""
        
    def get_decomposition_hints(self, qtype: str) -> str:
        """Get type-specific decomposition hints for LLM"""
```

### **Phase 3: Update Sequential Answering**

**File:** `sequential_answering.py`

**Changes:**
1. Support both placeholder syntaxes: `[SQ{N}_Answer]` and `#1`
2. Handle 3-4 hop questions (current supports 2-3)
3. Dynamic batching based on actual dependencies

```python
def substitute_answers(question: str, answers: Dict) -> str:
    """Support both [SQ1_Answer] and #1 syntax"""
    
    # Current: [SQ1_Answer], [SQ2_Answer]
    # Add: #1, #2, #3, #4
```

### **Phase 4: Evaluation on All Datasets**

**New file:** `run_multidataset_evaluation.py`

```python
async def evaluate_all_datasets():
    datasets = [
        ('HotpotQA/hotpotqa_sample_200.json', 'HotpotQA'),
        ('2WikiMultihopQA/2wikimultihopqa_sample_200.json', '2WikiMultihopQA'),
        ('MuSiQue/musique_qa_sample_200.json', 'MuSiQue')
    ]
    
    for dataset_path, dataset_name in datasets:
        results = await evaluate_dataset(dataset_path, dataset_name)
        save_results(results, f'Results/{dataset_name}_results.json')
```

---

## 📊 Expected Decomposition Patterns

### **Question Type Mapping**

| Dataset | Question Type | Typical Hops | Decomposition Strategy |
|---------|--------------|--------------|------------------------|
| HotpotQA | Bridge | 2 | Property lookup → Query |
| HotpotQA | Comparison | 2 | Parallel property extraction → Compare |
| 2Wiki | Compositional | 2-3 | Nested property lookups |
| 2Wiki | Bridge-Comparison | 3-4 | Bridge for both entities → Compare |
| 2Wiki | Inference | 2-3 | Relationship chain reasoning |
| MuSiQue | Mixed | 2-4 | Flexible multi-hop chains |

### **Decomposition Templates**

**Bridge (2-hop):**
```
Q: "A overlooks B. How long is B?"
SQ1: "What does A overlook?"
SQ2: "How long is [SQ1_Answer]?"
```

**Compositional (2-3 hop):**
```
Q: "Where was the director of film X born?"
SQ1: "Who directed film X?"
SQ2: "Where was [SQ1_Answer] born?"
```

**Bridge-Comparison (3-4 hop):**
```
Q: "Which film has the director who died earlier, A or B?"
SQ1: "Who directed film A?"
SQ2: "Who directed film B?"
SQ3: "When did [SQ1_Answer] die?"
SQ4: "When did [SQ2_Answer] die?"
SQ5: "Compare [SQ3_Answer] and [SQ4_Answer]"
```

**Inference (2-3 hop):**
```
Q: "Who is the father-in-law of X?"
SQ1: "Who is X's spouse?"
SQ2: "Who is the father of [SQ1_Answer]?"
```

**MuSiQue Property Chain (3-4 hop):**
```
Q: "What is the symbol of the team from the headquarters of the manufacturer of X?"
SQ1: "X >> manufacturer"
SQ2: "#1 >> headquarters location"
SQ3: "What is the symbol of the team from #2?"
```

---

## 🚀 Next Steps

### **Immediate (This Session):**
1. ✅ Analyze all datasets ← DONE
2. 🔄 Design unified decomposition strategy ← IN PROGRESS
3. ⏳ Implement enhanced `query_decomposition.py`
4. ⏳ Add support for new question types

### **Short-term:**
1. Update `sequential_answering.py` for 3-4 hop support
2. Test on 2WikiMultihopQA sample
3. Test on MuSiQue sample
4. Compare with gold decompositions (MuSiQue has them!)

### **Medium-term:**
1. Switch to `metadata_v3.db`
2. Integrate passage content
3. Optimize retrieval for compositional questions
4. Full evaluation on all 3 datasets

---

## 📝 Notes

### **MuSiQue Gold Decompositions**
- MuSiQue provides gold decompositions ✅
- Can be used for:
  - Training decomposition model
  - Validation of LLM decomposition
  - Benchmark comparison

### **2WikiMultihopQA Evidence**
- Has `evidences` field (which documents support the answer)
- Has `supporting_facts` field
- Can validate retrieval quality

### **Cross-Dataset Challenges**
1. **Placeholder syntax varies**: `[SQ1_Answer]` vs `#1`
2. **Decomposition granularity**: Some datasets have finer steps
3. **Question complexity**: MuSiQue has longest chains (up to 4 hops)
4. **Type diversity**: 2Wiki has most diverse question types

---

## ✅ Recommendation

**Use Approach 3 (Hybrid)**:
1. Flexible enough for all datasets
2. Pattern hints improve consistency
3. LLM handles edge cases
4. Validation ensures quality

**Start with:**
- Update `query_decomposition.py` prompt to be type-agnostic
- Support 2-4 hop questions
- Add `#N` placeholder support alongside `[SQ{N}_Answer]`
- Test on all 3 datasets

