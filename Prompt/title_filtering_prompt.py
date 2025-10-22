"""
Title Filtering Prompt
=======================
LLM-based filtering of candidate passages using title + metadata snippet.
Used in both Stage 1-A (Value matching) and Stage 1-B (Type filtering).
"""

TITLE_FILTERING_PROMPT = """You are an expert at identifying relevant passages for answering questions.

Given:
- A QUERY
- An ENTITY extracted from the query (with type/subtype)
- A list of CANDIDATE PASSAGES (with title + snippet from metadata)

Your task: Filter the candidates to keep only those that are RELEVANT to answering the query about the entity.

FILTERING CRITERIA:
1. **Direct match**: Title/snippet is about the entity or directly related concept
2. **Contextual relevance**: Provides information needed to answer the query
3. **Alias/spelling variations**: Consider "NHL" = "National Hockey League", "Roissy" = "Charles de Gaulle Airport"
4. **Remove unrelated**: Filter out passages that share the same type but are unrelated entities

**IMPORTANT RULES:**
- Be INCLUSIVE rather than exclusive - when in doubt, keep it
- **If ALL candidates are irrelevant, return EMPTY ARRAY []** - don't force-keep any
- Use both title AND snippet to make decisions (snippet shows key metadata attributes)

Examples of what to KEEP:
- Query: "airport named after Pat McCarran" → Keep: "McCarran International Airport" (direct match)
- Query: "2015 NHL Entry Draft" → Keep: "Connor McDavid" if snippet mentions "2015 NHL draft"
- Query: "ghost man slang" → Keep: "Gweilo" if snippet says "Cantonese slang for ghost man"

Examples of what to FILTER OUT:
- Query: "Argentine education" → Remove: "Education in Morocco" (same type, different country)
- Query: "Stephen Graham 2006 film" → Remove: "Stephen Wade" (different person, no 2006 film)
- Query: "Baltic Cup" → Remove: "FIFA World Cup" (same type, completely different event)

**ZERO RESULTS OK:** If no candidates are relevant, return `"relevant_titles": []`

---

**INPUT:**

Query: {{QUERY}}

Entity: {{ENTITY_NAME}}
Type: {{ENTITY_TYPE}}
Subtype: {{ENTITY_SUBTYPE}}

Candidate Passages ({{COUNT}} total):
{{CANDIDATES}}

---

**OUTPUT FORMAT (JSON only):**

{
  "relevant_titles": ["title1", "title2", ...],
  "filtered_out_titles": ["title3", "title4", ...],
  "reasoning": "Brief explanation of filtering decisions"
}

**Rules:**
- Include a title in "relevant_titles" if it could help answer the query
- Return EMPTY ARRAY [] if all candidates are irrelevant
- Be inclusive - when in doubt, keep it
- Provide concise reasoning for major filtering decisions
"""
