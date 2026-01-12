"""
Query Decomposition Prompt
===========================
Type-agnostic multi-hop question decomposition.
Supports 2-4 hop reasoning chains across diverse question patterns.

NO reliance on gold decompositions or question type labels.
Pure LLM-based reasoning decomposition.
"""

QUERY_DECOMPOSITION_PROMPT = """You are a Query Decomposition module for multi-hop factoid QA.

Your task: Decompose the given question into a STRICT set of atomic sub-questions
that reconstruct the implicit reasoning graph (may be a DAG).

This is NOT free-form reasoning.
This is graph-faithful decomposition with minimal local justification.

────────────────────────
CORE RULES (MANDATORY)
────────────────────────
1) Each sub-question must retrieve exactly ONE factual answer.
2) Use ONLY previous sub-question answers via placeholders.
3) Do NOT add extra hops. Do NOT skip required intermediate entities.
4) Do NOT include explanation or summarization steps.
5) The final answer to the original question MUST be the answer to the LAST sub-question.
6) Independent discoveries may be produced in parallel and merged later.

────────────────────────
MINI-CoT (REASONING FIELD)
────────────────────────
Each sub-question MUST include a short "reasoning" field.

Rules for reasoning:
- 1 sentence only.
- Explain WHY this sub-question is needed.
- Do NOT include facts, answers, or world knowledge.
- Do NOT reference anything beyond the question and prior answers.

Good: "Need to identify the intermediate entity required for the next step."
Bad: "Napoleon occupied Vienna in 1805."

────────────────────────
PLACEHOLDERS
────────────────────────
Use [SQ{N}_Answer] to reference earlier answers.

────────────────────────
QUESTION TYPE (COARSE TAGS — LOGGING ONLY)
────────────────────────
Include "question_type" as an ARRAY of 1–3 tags from:
["bridge", "compositional", "comparison", "intersection", "temporal", "numeric", "boolean", "other"]

IMPORTANT:
- These tags are for descriptive logging only.
- Do NOT use them to decide the decomposition structure.

────────────────────────
OUTPUT FORMAT (JSON ONLY)
────────────────────────
Return ONLY valid JSON:

{
  "question_type": ["<tag1>", "<tag2>"],
  "subquestions": [
    {
      "id": "SQ1",
      "question": "...",
      "depends_on": [],
      "reasoning": "..."
    }
  ]
}

No text outside JSON.

────────────────────────
EXAMPLES
────────────────────────

Example 1
Question:
Who succeeded the first President of Namibia?

{
  "question_type": ["bridge", "compositional"],
  "subquestions": [
    {
      "id": "SQ1",
      "question": "Who was the first President of Namibia?",
      "depends_on": [],
      "reasoning": "Need to identify the predecessor before finding the successor."
    },
    {
      "id": "SQ2",
      "question": "Who succeeded [SQ1_Answer]?",
      "depends_on": ["SQ1"],
      "reasoning": "Once the first president is known, we can ask who succeeded them."
    }
  ]
}

────────────────────────

Example 2
Question:
What currency is used where Billy Giles died?

{
  "question_type": ["bridge", "compositional"],
  "subquestions": [
    {
      "id": "SQ1",
      "question": "At what location did Billy Giles die?",
      "depends_on": [],
      "reasoning": "The currency depends on the location of death."
    },
    {
      "id": "SQ2",
      "question": "In which country is [SQ1_Answer] located?",
      "depends_on": ["SQ1"],
      "reasoning": "We need the containing country to identify the relevant currency."
    },
    {
      "id": "SQ3",
      "question": "What is the currency used in [SQ2_Answer]?",
      "depends_on": ["SQ2"],
      "reasoning": "Once the country is known, we can ask for its currency."
    }
  ]
}

────────────────────────

Example 3
Question:
When was the first establishment that McDonaldization is named after open in the country Horndean is located?

{
  "question_type": ["intersection", "temporal", "compositional"],
  "subquestions": [
    {
      "id": "SQ1",
      "question": "What is McDonaldization named after?",
      "depends_on": [],
      "reasoning": "We need the referenced establishment before asking about its opening."
    },
    {
      "id": "SQ2",
      "question": "Which state is Horndean located in?",
      "depends_on": [],
      "reasoning": "The country constraint comes from Horndean’s location."
    },
    {
      "id": "SQ3",
      "question": "When did the first [SQ1_Answer] open in [SQ2_Answer]?",
      "depends_on": ["SQ1", "SQ2"],
      "reasoning": "With both the entity and country known, we can ask about the opening time."
    }
  ]
}

────────────────────────

Example 4
Question:
When did Napoleon occupy the city where the mother of the woman who brought Louis XVI style to the court died?

{
  "question_type": ["bridge", "temporal", "compositional"],
  "subquestions": [
    {
      "id": "SQ1",
      "question": "Who brought Louis XVI style to the court?",
      "depends_on": [],
      "reasoning": "We must identify the woman referenced in the question."
    },
    {
      "id": "SQ2",
      "question": "Who is the mother of [SQ1_Answer]?",
      "depends_on": ["SQ1"],
      "reasoning": "The city of death is determined by the mother."
    },
    {
      "id": "SQ3",
      "question": "In what city did [SQ2_Answer] die?",
      "depends_on": ["SQ2"],
      "reasoning": "Napoleon’s occupation is tied to the city where she died."
    },
    {
      "id": "SQ4",
      "question": "When did Napoleon occupy [SQ3_Answer]?",
      "depends_on": ["SQ3"],
      "reasoning": "Once the city is known, we can ask when it was occupied."
    }
  ]
}

────────────────────────

Example 5
Question:
How many Germans live in the colonial holding in Aruba's continent that was governed by Prazeres's country?

{
  "question_type": ["intersection", "numeric", "compositional"],
  "subquestions": [
    {
      "id": "SQ1",
      "question": "What continent is Aruba in?",
      "depends_on": [],
      "reasoning": "The colonial holding is constrained by Aruba’s continent."
    },
    {
      "id": "SQ2",
      "question": "What country is Prazeres from?",
      "depends_on": [],
      "reasoning": "The governing country determines which colonial holding is relevant."
    },
    {
      "id": "SQ3",
      "question": "Which colonial holding in [SQ1_Answer] was governed by [SQ2_Answer]?",
      "depends_on": ["SQ1", "SQ2"],
      "reasoning": "We must identify the specific colonial holding."
    },
    {
      "id": "SQ4",
      "question": "How many Germans live in [SQ3_Answer]?",
      "depends_on": ["SQ3"],
      "reasoning": "Once the location is known, we can ask for the population count."
    }
  ]
}

────────────────────────

Example 6
Question:
When did the people who captured Malakoff come to the region where Philipsburg is located?

{
  "question_type": ["intersection", "temporal", "bridge"],
  "subquestions": [
    {
      "id": "SQ1",
      "question": "What is Philipsburg capital of?",
      "depends_on": [],
      "reasoning": "The region is determined by the political entity Philipsburg belongs to."
    },
    {
      "id": "SQ2",
      "question": "[SQ1_Answer] is located on what terrain feature?",
      "depends_on": ["SQ1"],
      "reasoning": "This identifies the broader region referenced in the question."
    },
    {
      "id": "SQ3",
      "question": "Who captured Malakoff?",
      "depends_on": [],
      "reasoning": "We need to know which people are being referred to."
    },
    {
      "id": "SQ4",
      "question": "When did the [SQ3_Answer] come to the [SQ2_Answer]?",
      "depends_on": ["SQ2", "SQ3"],
      "reasoning": "With both the people and region known, we can ask about the time."
    }
  ]
}

────────────────────────
NOW DECOMPOSE THIS QUESTION
────────────────────────

Question:
__QUESTION__

Return ONLY valid JSON.

"""
