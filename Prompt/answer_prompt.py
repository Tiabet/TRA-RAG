"""Answering Prompts
=================

This module consolidates all prompts used for:
- sub-question answering
- final (main-question) answer synthesis
- naive RAG (single-step) answering

It replaces the legacy split modules:
- Prompt/answer.py
- Prompt/subquestion_answering_prompt.py

Keep prompts here to avoid drift across pipelines.
"""

# ---------------------------------------------------------------------------
# Detailed Sub-Question Answering (Intermediate SQs)
# ---------------------------------------------------------------------------

DETAILED_SUBQUESTION_ANSWERING_PROMPT = """---Role---
You are a multi-hop retrieval-augmented assistant specializing in extracting precise information from provided passages.

---Main Query (Context)---
{{main_query}}

---Goal---
Carefully read ALL the Information passages (both current and previous context) and extract the answer to the Sub-Question.
Use ONLY the given Information. Be AGGRESSIVE in finding the answer - check every field, every sentence, every detail.

---Where to Look for Answers---
1. **Passage Titles**: Often contain the answer directly (e.g., "Believe (Cher song)")
2. **Description Field**: First place to check for detailed information
3. **Main_entity Field**: Key facts about the entity
4. **Attributes Fields**: Specific properties (dates, locations, measurements, etc.)
5. **Events Field**: Historical facts, timeline information, relationships
6. **Relations Field**: Connections to other entities
7. **All Other Metadata**: Any field might contain the answer!

---Instructions---
1. **Read ALL Passages Thoroughly**:
   - Check EVERY passage (current + previous context)
   - Look at EVERY metadata field, not just description
   - The answer might be buried in ANY field

2. **Start with Passage Titles**:
   - Titles often contain entity names directly
   - Example: "Believe (Cher song)" → if asked about Cher's song, the title IS the answer

3. **Scan All Metadata Fields**:
   - description, main_entity, attributes, events, relations, etc.
   - Dates in: publication_date, birth_date, founded, active_years
   - Names in: main_entity, creator, director, members, related_entities
   - Locations in: location, headquarters, country, region

4. **Extract Precisely**:
   - Find exact values: names, dates, numbers, places
   - Look for keywords from the question in ALL fields
   - Match question terms to metadata field names

5. **Apply Simple Reasoning When Needed**:
   - You CAN perform basic reasoning (arithmetic, temporal logic)
   - Example: "seven years before 1999" → calculate 1992
   - Example: "older brother of X" + "X born in 1980" → infer relationship
   - BUT only reason about information IN the passages

6. **Check Previous Context FIRST**:
   - Previous sub-question answers may already contain what you need
   - Previous passages are just as important as current ones

7. **Be Specific**:
   - Provide exact names, dates, places, or values
   - NOT generic: "a plan", "a person", "a location"
   - Extract EXACT values from metadata

---Response Format---
- SHORT, DIRECT answer (typically 1-10 words)
- Output ONLY the answer text (no extra explanation, no quotation marks)
- Do NOT include emojis, bullet points, or any decorative characters
- Use the same language as the Sub-Question
- For names: Use full names if available (e.g., "Ferdinand Magellan")
- For dates: Be precise (e.g., "1998" not "late 1990s")
- For lists: Include all relevant items (e.g., "Estonia, Latvia, Lithuania")

---Output Strictness (IMPORTANT)---
- Output ONLY the answer text.
- Do NOT include emojis, extra commentary, explanations, quotes, bullet points, or prefixes (e.g., "Answer:").

---CRITICAL: When to Say "Insufficient information"---
ONLY respond "Insufficient information." if ALL of the following are true:
✗ You checked EVERY passage (current + previous context)
✗ You checked EVERY metadata field (title, description, main_entity, attributes, events, relations, ALL others)
✗ You looked for keywords from the question in ALL fields
✗ You cannot find the information OR derive it through simple reasoning
✗ The passages genuinely do NOT contain the needed information

If you find PARTIAL information or RELATED information, provide the closest match from the passages.

---Previous Context (IMPORTANT - May contain the answer!)---
{{previous_context}}

---Current Information Passages---
{{passages}}

---Sub-Question---
{{subquestion}}

---Answer---
CHECK THOROUGHLY:
1. All passage TITLES (the answer might be in the title!)
2. ALL metadata fields (description, main_entity, attributes, events, relations, etc.)
3. Both CURRENT passages AND PREVIOUS context
4. Look for exact matches of question keywords in field names and values

Extract and provide ONLY the answer (max 15 words).
You can perform simple reasoning on passage information.
ONLY respond "Insufficient information." if you truly checked EVERYTHING and cannot find/derive the answer.
"""


FINAL_SUBQUESTION_ANSWERING_PROMPT = """---Role---
You are a multi-hop retrieval-augmented assistant.

---Main Query (Context)---
{{main_query}}

---Goal---
Read ALL the Information passages (both current and previous context) and generate the correct answer to the Sub-Question.
Be AGGRESSIVE in finding the answer - check passage titles, all metadata fields, and previous context.

---Where to Look---
1. **Passage Titles** - Often contain the answer directly
2. **Description Field** - Primary source of information
3. **ALL Metadata Fields** - attributes, events, relations, publication_date, etc.
4. **Previous Context** - May already have the answer

---Target response length and format---
- One-word or minimal-phrase answer (max 5 words).

---Output Constraint (Strict)---
- Output ONLY the answer text.
- Do NOT include emojis, bullet points, or any extra commentary.

---Response Rules---
✓ Check passage TITLES first (answer might be in the title!)
✓ Check ALL metadata fields thoroughly (description, main_entity, attributes, events, relations, etc.)
✓ Check BOTH current passages AND previous context passages
✓ You CAN perform simple reasoning (arithmetic, temporal logic, relationship inference)
✓ Example: "2300 - 1000" → calculate 1300
✓ If you find the information anywhere in passages, USE IT
✓ Output ONLY the answer text (no emojis, no extra words)
✗ Answer must be short and concise
✗ Answer language must match the Sub-Question language
✗ Do NOT use external knowledge not in passages
✗ ONLY respond "Insufficient information." if you checked EVERY passage, EVERY field, and truly cannot find/derive the answer

---Previous Context (IMPORTANT - May contain the answer!)---
{{previous_context}}

---Current Information---
{{passages}}

---Sub-Question---
{{subquestion}}

---Answer---
CHECK: 1) Passage titles, 2) All metadata fields, 3) Previous context, 4) Current passages
You can perform simple reasoning on passage information.
Provide only the answer (max 5 words). 
ONLY respond "Insufficient information." if you checked EVERYTHING and genuinely cannot find/derive the answer.
"""


# ---------------------------------------------------------------------------
# Short Sub-Question Answering (legacy/simple)
# ---------------------------------------------------------------------------

SUBQUESTION_ANSWERING_PROMPT = """---Role---
You are a multi-hop retrieval-augmented assistant.

---Goal---
Read the Information passages and generate the correct answer to the Sub-Question.
Use only the given Information; if it is insufficient, reply with "Insufficient information.".

---Target response length and format---
- One-word or minimal-phrase answer (max 5 words).

---Output Constraint (Strict)---
- Output ONLY the answer text.
- Do NOT include emojis, bullet points, or any extra commentary.

---Response Rules---
- Answer must be short and concise.
- Answer language must match the Sub-Question language.
- Do NOT add or invent facts beyond the Information.
- Output ONLY the answer text (no emojis, no extra commentary, no formatting, no prefix like "Answer:").
- If the Information does not contain the answer, respond with "Insufficient information." only.

---Previous Context---
{{previous_context}}

---Information---
{{passages}}

---Sub-Question---
{{subquestion}}

---Answer---
Provide only the answer (max 5 words). If information is insufficient, respond "Insufficient information.".
"""


# ---------------------------------------------------------------------------
# Final Answer Synthesis
# ---------------------------------------------------------------------------

FINAL_ANSWER_SYNTHESIS_PROMPT = """---Role---
You are a multi-hop retrieval-augmented assistant.

---Goal---
Read the Sub-Question Chain and All Retrieved Passages to generate the correct answer to the Main Query.
Use BOTH sub-question answers AND the original passages to verify and complete your answer.
Be AGGRESSIVE in finding the answer - even if a sub-question failed, the passages might still contain the answer.

---Critical Instructions---
1. **Check Sub-Question Answers**: See what information was already extracted
2. **Verify with Passages**: Cross-check answers against original passages
3. **Fill Gaps**: If any sub-question answered "Insufficient information", check if the passages actually contain that information
4. **Combine Information**: Synthesize answers from multiple sub-questions if needed
5. **Perform Simple Reasoning**: You CAN do arithmetic, temporal logic, relationship inference

---Success Strategy---
- If a sub-question failed ("Insufficient information"), DON'T give up!
- Re-examine the passages - the information might be there
- Look in passage TITLES and ALL metadata fields
- Combine partial information from multiple passages

---Target response length and format---
- One-word or minimal-phrase answer (max 5 words).

---Output Constraint (Strict)---
- Output ONLY the answer text.
- Do NOT include emojis, bullet points, or any extra commentary.

---Response Rules---
✓ Use BOTH sub-question answers AND passages
✓ Re-check passages if sub-questions failed
✓ Look at passage TITLES - they often contain key information
✓ Check ALL metadata fields (description, main_entity, attributes, events, relations, etc.)
✓ You CAN perform simple reasoning (e.g., "seven years before 1999" = 1992)
✓ Answer must be short and concise
✓ Answer language must match the Query language
✓ If you need yes/no, use "Yes" or "No" only
✗ Do NOT use external knowledge not present in passages or sub-answers
✗ ONLY respond "Insufficient information." if BOTH sub-answers AND passages lack the information

---Sub-Question Chain---
{{subquestion_chain}}

---All Retrieved Passages (from all sub-questions)---
{{passages}}

---Main Query---
{{main_question}}

---Final Answer---
CHECK: 1) Sub-question answers, 2) Passage titles, 3) All metadata fields, 4) Previous context
Even if sub-questions failed, re-examine passages for the answer.
You can perform simple reasoning on the information.
Provide only the answer (max 5 words). 
ONLY respond "Insufficient information." if both sub-answers AND passages truly lack the needed information.
"""


# ---------------------------------------------------------------------------
# Naive RAG (single-step)
# ---------------------------------------------------------------------------

NAIVE_RAG_FINAL_ANSWER_PROMPT = """---Role---
You are a retrieval-augmented assistant.

---Goal---
Read the Retrieved Passages to generate the correct answer to the Question.
Use ONLY the given passages; do NOT use external knowledge.

---Critical Instructions---
1. Read ALL provided passages carefully
2. Extract relevant facts from the passages
3. If multiple passages are needed, combine them
4. Perform simple reasoning if required (arithmetic, temporal logic, comparisons)

---Target response length and format---
- One-word or minimal-phrase answer (max 5 words).

---Output Constraint (Strict)---
- Output ONLY the answer text.
- Do NOT include emojis, bullet points, or any extra commentary.

---Response Rules---
✓ Use ONLY the information provided in the passages
✓ Answer must be short and concise
✓ Answer language must match the Question language
✗ Do NOT hallucinate or invent facts
✗ ONLY respond "Insufficient information." if passages truly lack the needed information

---Retrieved Passages---
{{passages}}

---Question---
{{question}}

---Answer---
Provide only the answer (max 5 words).
"""
