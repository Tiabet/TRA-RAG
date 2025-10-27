# Project Structure

## 📁 Main Directory Structure

```
ChunkRAG_v2/
├── 📂 Core Modules (Root Level)
│   ├── multihop_pipeline.py          # Main pipeline orchestrator
│   ├── query_decomposition.py        # Query decomposition module
│   ├── sequential_answering.py       # Sequential sub-question answering
│   ├── hybrid_retrieval.py           # Hybrid retrieval system
│   ├── metadata_db.py                # Metadata database interface
│   ├── llm_logger.py                 # LLM interaction logging
│   ├── llm_evaluation.py             # LLM-based evaluation
│   └── judge_F1.py                   # Token-based F1 evaluation
│
├── 📂 Prompts/                       # All prompt templates
│   ├── answer_short_v2.py           # Answer generation prompts
│   ├── entity_extraction_prompt.py  # Entity extraction
│   ├── llm_filtering_prompt.py      # LLM filtering
│   ├── metadata_construction_prompt.py
│   ├── title_filtering_prompt.py
│   └── type_schema.py
│
├── 📂 Results/                       # All experiment results
│   ├── multihop_pipeline_200_results.json
│   ├── multihop_pipeline_200_results_IMPROVED.json
│   ├── llm_evaluation_results.json
│   ├── llm_evaluation_thrag_results.json
│   ├── evaluation_results_summary.json
│   ├── improved_evaluation_metrics.json
│   ├── thrag_evaluation_metrics.json
│   ├── still_wrong_cases.json
│   └── 📂 Logs/                      # LLM interaction logs
│       └── llm_log_*.txt
│
├── 📂 Analysis/                      # Analysis scripts and reports
│   ├── analyze_metadata_length.py
│   ├── analyze_results.py
│   ├── compare_with_thrag.py
│   ├── evaluate_retrieval_recall.py
│   ├── IMPROVED_PROMPT_SUMMARY.md
│   ├── LLM_FILTERING_IMPROVEMENTS.md
│   ├── PROJECT_SUMMARY.md
│   └── SQ_ANSWER_GENERATION_METADATA_ANALYSIS.md
│
├── 📂 Scripts/                       # Utility scripts
│   ├── extract_still_wrong.py
│   └── merge_improved_results.py
│
├── 📂 Dataset Folders/
│   ├── 📂 HotpotQA/                  # Main dataset
│   │   ├── hotpotqa_sample_200.json
│   │   ├── qa.json
│   │   ├── metadata_v2.db           # Metadata database
│   │   └── hotpotqa_sample_200_metadata_v2.json
│   │
│   ├── 📂 2WikiMultihopQA/
│   ├── 📂 MuSiQue/
│   ├── 📂 THRAG/                     # Comparison baseline
│   │   └── hotpot_30_5_filtered_200.json
│   └── 📂 NaiveRAG/                  # Naive RAG baseline
│
└── 📂 tests_archive/                 # Archived test files
    ├── test_multihop_200.py
    ├── test_improved_prompt.py
    └── ...
```

## 🔧 Core Modules

### Main Pipeline
- **multihop_pipeline.py**: Orchestrates the entire multi-hop QA pipeline
- **query_decomposition.py**: Decomposes complex queries into sub-questions
- **sequential_answering.py**: Answers sub-questions sequentially
- **hybrid_retrieval.py**: Combines different retrieval strategies

### Database & Logging
- **metadata_db.py**: SQLite-based metadata storage and retrieval
- **llm_logger.py**: Logs all LLM interactions for debugging
  - Logs saved to: `Results/Logs/llm_log_YYYYMMDD_HHMMSS.txt`

### Evaluation
- **llm_evaluation.py**: LLM-based answer evaluation (using GPT-4o-mini)
- **judge_F1.py**: Token-based evaluation (EM, F1, Precision, Recall, Accuracy)

## 📊 Results Organization

### Results/
All experimental outputs are stored here:
- **Pipeline outputs**: `multihop_pipeline_200_results*.json`
- **Evaluation results**: `*_evaluation_*.json`
- **Metrics**: `*_metrics.json`
- **Error analysis**: `still_wrong_cases*.json`

### Results/Logs/
All LLM interaction logs with timestamps:
- Format: `llm_log_YYYYMMDD_HHMMSS.txt`
- Contains: Full prompts, responses, and metadata for debugging

## 📈 Analysis Folder

### Analysis Scripts
- `analyze_*.py`: Various analysis scripts
- `compare_*.py`: Comparison scripts (e.g., vs THRAG)
- `evaluate_*.py`: Evaluation scripts

### Analysis Reports (Markdown)
- `IMPROVED_PROMPT_SUMMARY.md`: Prompt improvement results
- `LLM_FILTERING_IMPROVEMENTS.md`: LLM filtering analysis
- `PROJECT_SUMMARY.md`: Overall project summary
- `SQ_ANSWER_GENERATION_METADATA_ANALYSIS.md`: Sub-question answering analysis

## 🗂️ Dataset Structure

### HotpotQA/ (Main Dataset)
- `hotpotqa_sample_200.json`: 200 multi-hop questions
- `qa.json`: Gold answer mapping
- `metadata_v2.db`: Pre-built metadata database
- `hotpotqa_sample_200_metadata_v2.json`: Metadata JSON backup

### Other Datasets
- **2WikiMultihopQA/**: Alternative multi-hop dataset
- **MuSiQue/**: MuSiQue dataset
- **THRAG/**: Comparison baseline results
- **NaiveRAG/**: Naive RAG baseline results

## 🧪 Testing

All test files are archived in `tests_archive/`:
- Unit tests for specific components
- Integration tests
- Debug scripts

## 🚀 Quick Start

### Run Main Pipeline
```bash
python multihop_pipeline.py
```

### Evaluate Results (Token-based)
```bash
python judge_F1.py --pred Results/multihop_pipeline_200_results.json --gold HotpotQA/qa.json
```

### Evaluate Results (LLM-based)
```bash
python llm_evaluation.py --pred Results/multihop_pipeline_200_results.json --gold HotpotQA/qa.json --out Results/llm_eval_output.json
```

### Compare with THRAG
```bash
python Analysis/compare_with_thrag.py
```

## 📝 Configuration

Environment variables (`.env`):
```
ALICE_OPENAI_KEY=your_api_key
ALICE_CHAT_URL=your_api_url
```

## 🔍 Key File Paths in Code

All code now uses standardized paths:
- **Results**: `Results/`
- **Logs**: `Results/Logs/`
- **Dataset**: `HotpotQA/`
- **Metadata DB**: `HotpotQA/metadata_v2.db`

## 📦 Dependencies

See `requirements.txt` for Python package dependencies.
