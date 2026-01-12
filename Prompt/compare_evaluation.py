EVALUATION_PROMPT = """
### Role
You are an expert in Knowledge Graph construction and Natural Language Processing. Your task is to rigorously evaluate and compare the quality of two candidate structured extractions produced from a given source text (Corpus).

Notes:
- Candidate 1 may be a list of Triplets or a list of HyperEdges (or similar extracted structure).
- Candidate 2 is a metadata-derived structure (ERA-style) for the same corpus.

### Evaluation Rubrics
Score each candidate on a scale of 1 to 5 for the following criteria:
1. **Completeness**: Does the set capture all key facts, entities, and relations present in the corpus? Are there any significant omissions?
2. **Faithfulness**: Are the triplets strictly supported by the corpus? Is there any hallucination, exaggeration, or distortion of facts?
3. **Structural Clarity**: Are subjects and objects clear noun phrases? Are predicates well-defined? Are pronouns (e.g., "he", "it") resolved to their specific entities?
4. **Atomic Granularity**: Is each triplet a single, irreducible fact? Are complex sentences properly decomposed into multiple simple triplets?

### Execution Instructions
1. **Analyze the Corpus**: First, list the core factual points that must be extracted from the provided text.
2. **Evaluate Candidate 1 & 2**: Analyze each candidate against the rubrics. You must explicitly identify missing information or incorrect relations.
3. **Justification**: Provide brief, evidence-grounded justification for your scores (no long reasoning).
4. **Final Verdict**: Select the winner based on the average scores.

---
### Input Data
- **Original Corpus**: 
{{original_corpus}}

- **Candidate 1**: 
{{triplet}}

- **Candidate 2**: 
{{ERA_structure}}

---
### Output Format (STRICT JSON ONLY)
Return a single JSON object with this exact schema:

{
	"scores": {
		"completeness": {"candidate1": 1, "candidate2": 1},
		"faithfulness": {"candidate1": 1, "candidate2": 1},
		"structural_clarity": {"candidate1": 1, "candidate2": 1},
		"atomic_granularity": {"candidate1": 1, "candidate2": 1}
	},
	"average": {"candidate1": 1.0, "candidate2": 1.0},
	"final_decision": "Candidate 1",
	"brief_justification": {
		"candidate1": "...",
		"candidate2": "..."
	}
}

Constraints:
- Scores are integers 1..5.
- "final_decision" is exactly "Candidate 1" or "Candidate 2".
- Output JSON only (no markdown, no extra text).
"""