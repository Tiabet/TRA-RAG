"""Prompts for QD_v2.

We use a single-call prompt to reduce latency:
- The model answers the sub-question AND selects which documents it used.

The pipeline expects strict JSON output.
"""


SUBQUESTION_ANSWERING_WITH_SELECTION_PROMPT = """You will be given:
- a sub-question
- a list of numbered documents (each has title + passage text)

Task:
1) Answer the sub-question as briefly and directly as possible.
2) Select the MINIMAL set of document numbers that are actually necessary to justify your answer.

Rules:
- If the answer is 'Insufficient information.' (or equivalent), return an empty list for needed_docs.
- Only select documents that contain direct evidence.
- Prefer fewer documents.

Output JSON ONLY with this schema:
{
  "answer": "...",
  "needed_docs": [1, 3],
  "notes": "short reason"
}

Sub-question:
{{subquestion}}

Documents:
{{documents}}
"""


# Backwards-compat alias (kept to avoid breaking older imports).
PASSAGE_SELECTION_PROMPT = SUBQUESTION_ANSWERING_WITH_SELECTION_PROMPT