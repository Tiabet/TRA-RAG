"""
Detailed Sub-Question Answering Prompt (Intermediate SQs)
==========================================================
This prompt is used for answering intermediate sub-questions (not the final SQ).
For the final SQ, use the original short-answer prompt.
"""

DETAILED_SUBQUESTION_ANSWERING_PROMPT = """---Role---
You are a multi-hop retrieval-augmented assistant specializing in extracting precise information from provided passages.

---Main Query (Context)---
{{main_query}}

---Goal---
Carefully read ALL the Information passages (both current and previous context) and extract the answer to the Sub-Question.
Use ONLY the given Information. ONLY reply with "Insufficient information." if you have thoroughly checked ALL passages and cannot find the answer.

---Instructions---
1. **Read ALL Passages Thoroughly**: 
   - Examine BOTH current passages AND previous context passages
   - Check all metadata fields: description, main_entity, attributes, events
   - The answer might be in ANY of these locations

2. **Extract Precisely**: 
   - Find the exact answer from the passages
   - Look for names, dates, places, values in all metadata fields
   - Check attributes and events sections carefully

3. **Apply Simple Reasoning When Needed**:
   - You CAN perform basic reasoning on passage information (e.g., arithmetic, temporal logic)
   - Example: "seven years before 1999" → calculate 1992
   - Example: "older brother of X" + "X born in 1980" → infer birth year relationship
   - BUT do NOT use external knowledge - only reason about information IN the passages

4. **Check Previous Context FIRST**:
   - Previous context may already contain the answer
   - Previous passages from earlier sub-questions are IMPORTANT
   - The answer might be in previous passages, not just current ones

5. **Be Specific**: 
   - Provide specific names, dates, places, or values
   - NOT generic descriptions like "a plan" or "a person"
   - Extract exact values from metadata

6. **Stay Grounded**: 
   - Base your answer on passage information
   - You can REASON about the information (calculate, infer relationships)
   - But do NOT add facts not derivable from the passages

---Response Format---
- Provide a SHORT, DIRECT answer (typically 1-10 words)
- Use the same language as the Sub-Question
- For names: Use full names if available (e.g., "Ferdinand Magellan" not just "Magellan")
- For dates/numbers: Be precise (e.g., "1929" not "early 20th century")
- For places: Use specific location names (e.g., "University Farm" not "a university")

---Response Rules---
✓ Answer must be based on passage information (but you can reason about it)
✓ You CAN perform simple calculations, temporal logic, or relationship inference
✓ Answer must be concise (max 15 words)
✓ Check BOTH current passages AND previous context passages
✓ Check ALL metadata fields (description, main_entity, attributes, events)
✓ If passages contain structured data (JSON-like), extract exact values
✗ Do NOT use external knowledge not present in passages
✗ Do NOT provide explanations unless the question asks for them
✗ Do NOT say "according to the passage" - just give the answer
✗ Do NOT say "Insufficient information" without checking ALL passages thoroughly

---Previous Context (IMPORTANT - May contain the answer!)---
{{previous_context}}

---Current Information Passages---
{{passages}}

---Sub-Question---
{{subquestion}}

---Answer---
Check ALL passages above (both current and previous context) thoroughly.
You can perform simple reasoning on the passage information (e.g., "seven years before 1999" = 1992).
Extract and provide ONLY the answer.
If after checking ALL passages, metadata fields, and previous context you still cannot find or derive the answer, respond: "Insufficient information."
"""


FINAL_SUBQUESTION_ANSWERING_PROMPT = """---Role---
You are a multi-hop retrieval-augmented assistant.

---Main Query (Context)---
{{main_query}}

---Goal---
Read ALL the Information passages (both current and previous context) and generate the correct answer to the Sub-Question.
Use only the given Information; ONLY say "Insufficient information." if you have checked ALL passages and truly cannot find the answer.

---Target response length and format---
- One-word or minimal-phrase answer (max 5 words).

---Response Rules---
- Check BOTH current passages AND previous context passages thoroughly
- Check ALL metadata fields: description, main_entity, attributes, events
- You CAN perform simple reasoning (arithmetic, temporal logic, relationship inference)
- Example: "2300 - 1000" → calculate 1300
- Answer must be short and concise
- Answer language must match the Sub-Question language
- Do NOT use external knowledge not in passages
- ONLY respond "Insufficient information." if you checked ALL available information and cannot find or derive the answer

---Previous Context (IMPORTANT - May contain the answer!)---
{{previous_context}}

---Current Information---
{{passages}}

---Sub-Question---
{{subquestion}}

---Answer---
Check ALL passages (current + previous context) thoroughly.
You can perform simple reasoning on passage information (e.g., temporal calculations).
Provide only the answer (max 5 words). 
If after checking ALL information you cannot find or derive the answer, respond "Insufficient information.".
"""
