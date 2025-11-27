"""
Title Filtering Prompt
=======================
LLM-based filtering of candidate passages using title + metadata snippet.
Used in both Stage 1-A (Value matching) and Stage 1-B (Type filtering).
"""

TITLE_FILTERING_PROMPT = """You are an expert at identifying relevant passages for answering questions.

Given:
- A MAIN QUERY (the original multi-hop question)
- A SUB-QUERY (the current sub-question being answered)
- An ENTITY extracted from the sub-query (with type/subtype)
- A list of CANDIDATE PASSAGES with:
  * Title
  * Type/Subtype
  * Matched Content (the actual key-value pairs from metadata that matched the search)

Your task: Filter the candidates to keep only those that are RELEVANT to:
1. Answering the SUB-QUERY about the entity
2. Contributing to answering the MAIN QUERY

**CRITICAL: Use ONLY the information provided in the Matched Content. Do NOT use your internal knowledge about these topics.**

FILTERING CRITERIA:
1. **Direct match**: Title + Matched Content shows this passage is about the entity or directly related concept
2. **Sub-Query relevance**: Matched Content provides information needed to answer the current sub-question
3. **Main Query relevance**: Matched Content contributes to the overall main question
4. **Alias/spelling variations**: Consider variations visible in the Matched Content
5. **Remove unrelated**: Filter out passages where Matched Content shows unrelated entities (same type but different topic)

**IMPORTANT RULES:**
- **DO NOT use your background knowledge** - judge relevance ONLY from Title + Matched Content
- **A passage must be relevant to BOTH the Sub-Query AND contribute to the Main Query**
- If Matched Content is irrelevant to the Main Query AND unhelpful for the Sub-Query, REMOVE it
- Be INCLUSIVE rather than exclusive - when in doubt, keep it
- **If ALL candidates are irrelevant, return EMPTY ARRAY []** - don't force-keep any
- The Matched Content shows the ACTUAL key-value pairs that matched - use this to judge relevance

Examples:
- If Matched Content shows "description: A project for the restructuring of higher education in Argentina" → Relevant for "Argentine education plan"
- If Matched Content shows "location: Morocco" → NOT relevant for "Argentine education plan"
- If Matched Content shows "full_name: Taquini Plan, alternative_name: Plan de Creación de Nuevas Universidades" → Relevant for "Who proposed plan"

**ZERO RESULTS OK:** If no candidates are relevant to BOTH queries based on their Matched Content, return `"relevant_titles": []`

---

**INPUT:**

Main Query: {{MAIN_QUERY}}

Sub-Query: {{SUB_QUERY}}

Entity: {{ENTITY_NAME}}
Type: {{ENTITY_TYPE}}
Subtype: {{ENTITY_SUBTYPE}}

Candidate Passages ({{COUNT}} total):
{{CANDIDATES}}

Each candidate shows:
- title: The passage title
- type/subtype: The metadata classification
- matched_content: The ACTUAL key-value pairs from metadata that matched your search
  (This is what you should base your decision on - NOT your internal knowledge)

---

**OUTPUT FORMAT (JSON only):**

{
  "relevant_titles": ["title1", "title2", ...],
  "filtered_out_titles": ["title3", "title4", ...],
  "reasoning": "Brief explanation based on the Matched Content shown (not internal knowledge)"
}

**Rules:**
- Judge relevance ONLY from the provided Matched Content
- Include a title in "relevant_titles" if its Matched Content could help answer the query
- Return EMPTY ARRAY [] if all candidates' Matched Content is irrelevant
- Be inclusive - when in doubt, keep it
- Provide reasoning that references the Matched Content you saw
"""
