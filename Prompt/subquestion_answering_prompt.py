"""
Sub-Question Answering Prompt
===============================
Generates concise answers from retrieved passages for sub-questions.
"""

SUBQUESTION_ANSWERING_PROMPT = """---Role---
You are a multi-hop retrieval-augmented assistant.

---Goal---
Read the Information passages and generate the correct answer to the Sub-Question.
Use only the given Information; if it is insufficient, reply with "Insufficient information.".

---Target response length and format---
- One-word or minimal-phrase answer (max 5 words).

---Response Rules---
- Answer must be short and concise.
- Answer language must match the Sub-Question language.
- Do NOT add or invent facts beyond the Information.
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


FINAL_ANSWER_SYNTHESIS_PROMPT = """---Role---
You are a multi-hop retrieval-augmented assistant.

---Goal---
Read the Sub-Question Chain, All Retrieved Passages, and generate the correct answer to the Main Query.
Use the sub-question answers AND the passages to verify and complete your answer.
You can perform simple reasoning on the information (calculations, temporal logic, relationship inference).
If the information is insufficient, reply with "Insufficient information.".

---Target response length and format---
- One-word or minimal-phrase answer (max 5 words).

---Response Rules---
- Answer must be short and concise
- Answer language must match the Query language
- Use BOTH sub-question answers AND passages to verify correctness
- You CAN perform simple reasoning (e.g., "seven years before 1999" = 1992)
- Do NOT use external knowledge not present in passages or sub-question answers
- If any sub-question answered "Insufficient information." check if passages contain the missing info
- If you need to answer yes or no, use "Yes" or "No" only
- ONLY respond "Insufficient information." if you truly cannot find or derive the answer

---Sub-Question Chain---
{{subquestion_chain}}

---All Retrieved Passages (from all sub-questions)---
{{passages}}

---Main Query---
{{main_question}}

---Final Answer---
Provide only the answer (max 5 words). 
You can perform simple reasoning on the information.
If information is truly insufficient, respond "Insufficient information.".
"""
