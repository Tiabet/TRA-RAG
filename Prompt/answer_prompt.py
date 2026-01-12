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
# Short Sub-Question Answering (legacy/simple)
# ---------------------------------------------------------------------------

SUBQUESTION_ANSWERING_PROMPT = """---Role---
You are a multi-hop retrieval-augmented assistant.

---Goal---
Read the Information passages and generate the correct answer to the Sub-Question.

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
- You should think step-by-step to answer the question, but output ONLY the final answer.

---Previous Context---
{{previous_context}}

---Information---
{{passages}}

---Sub-Question---
{{subquestion}}

- You should think step-by-step to answer the question, but output ONLY the final answer.

---Answer---

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
6. **Think Step-by-Step**: Carefully reason through the information before answering

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

**Think Step-by-Step**: Carefully reason through all information provided before answering

---Final Answer---

"""

# ---------------------------------------------------------------------------
# Naive RAG (single-step)
# ---------------------------------------------------------------------------

NAIVE_RAG_ANSWER_PROMPT = """---Role---
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

- You should think step-by-step to answer the question, but output ONLY the final answer.

---Answer---

"""