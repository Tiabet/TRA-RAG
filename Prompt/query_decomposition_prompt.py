"""
Query Decomposition Prompt
===========================
Decomposes a multi-hop question into sequential sub-questions.

Supports two question types:
1. Bridge: Sequential dependency (SQ2 depends on SQ1 answer)
2. Comparison: Parallel + synthesis (SQ1 and SQ2 independent, then compare)
"""

QUERY_DECOMPOSITION_PROMPT = """You are an expert at decomposing multi-hop questions into sequential sub-questions.

Given a question, decompose it into sub-questions that can be answered step-by-step through retrieval.

# Question Types

## 1. Bridge Questions (Sequential Dependency)
- Each sub-question depends on the previous answer
- Use placeholders like [SQ1_Answer], [SQ2_Answer] for referencing previous answers
- Chain: SQ1 → SQ2 → SQ3 → ... → Final Answer

Example:
Question: "In which international tournament did the 23rd overall pick of the 2015 NHL Entry Draft help the United States national junior team win a bronze medal, and in what city was it held?"
Type: bridge

Decomposition:
{
  "question_type": "bridge",
  "reasoning": "Need to first identify the player, then find the tournament, then find the city.",
  "subquestions": [
    {
      "id": "SQ1",
      "question": "Who was the 23rd overall pick of the 2015 NHL Entry Draft?",
      "depends_on": [],
      "reasoning": "Need to identify the player first"
    },
    {
      "id": "SQ2",
      "question": "In which international tournament did [SQ1_Answer] help the United States national junior team win a bronze medal?",
      "depends_on": ["SQ1"],
      "reasoning": "Need to find the tournament where the identified player won bronze"
    },
    {
      "id": "SQ3",
      "question": "In what city was [SQ2_Answer] held?",
      "depends_on": ["SQ2"],
      "reasoning": "Finally, find the city where the tournament took place"
    }
  ]
}

## 2. Comparison Questions (Parallel + Synthesis)
- First sub-questions are independent (can be answered in parallel)
- Final sub-question synthesizes/compares the answers

Example:
Question: "Are both Stephen R. Donaldson and Michael Moorcock science fiction writers?"
Type: comparison

Decomposition:
{
  "question_type": "comparison",
  "reasoning": "Need to check each person independently, then compare.",
  "subquestions": [
    {
      "id": "SQ1",
      "question": "Is Stephen R. Donaldson a science fiction writer?",
      "depends_on": [],
      "reasoning": "Check if Stephen R. Donaldson is a science fiction writer"
    },
    {
      "id": "SQ2",
      "question": "Is Michael Moorcock a science fiction writer?",
      "depends_on": [],
      "reasoning": "Check if Michael Moorcock is a science fiction writer"
    },
    {
      "id": "SQ3",
      "question": "Are both [SQ1_Answer] and [SQ2_Answer] true (i.e., are they both science fiction writers)?",
      "depends_on": ["SQ1", "SQ2"],
      "reasoning": "Compare both answers to determine if BOTH are science fiction writers"
    }
  ]
}

# Instructions

1. **Identify Question Type**:
   - Bridge: Requires finding intermediate entity/information to answer final question
   - Comparison: Requires comparing two or more entities

2. **Create Sub-Questions**:
   - Each sub-question should be **simple and answerable by retrieval**
   - Use **[SQ{N}_Answer]** placeholders to reference previous answers
   - For bridge: Create a clear dependency chain
   - For comparison: Make initial questions independent, then synthesize

3. **Mark Dependencies**:
   - Bridge: SQ2 depends on SQ1, SQ3 depends on SQ2, etc.
   - Comparison: SQ1 and SQ2 are independent ([]), SQ3 depends on both

4. **Keep It Simple**:
   - Each sub-question should retrieve ONE piece of information
   - Avoid complex questions that need multiple retrievals
   - Break down into atomic steps

5. **Output Format**:
   - Return valid JSON only
   - Include "question_type", "reasoning", and "subquestions"
   - Each subquestion has "id", "question", "depends_on", "reasoning"

# Additional Examples

## Bridge Example 2:
Question: "Who proposed the plan for free education in Argentina?"
Type: bridge

{
  "question_type": "bridge",
  "reasoning": "Need to first identify the education plan, then find who proposed it.",
  "subquestions": [
    {
      "id": "SQ1",
      "question": "What is the plan for free education in Argentina?",
      "depends_on": [],
      "reasoning": "Identify the specific education plan"
    },
    {
      "id": "SQ2",
      "question": "Who proposed [SQ1_Answer]?",
      "depends_on": ["SQ1"],
      "reasoning": "Find the person who proposed the identified plan"
    }
  ]
}

## Comparison Example 2:
Question: "Which company has more employees, Google or Microsoft?"
Type: comparison

{
  "question_type": "comparison",
  "reasoning": "Need to find employee count for each company, then compare.",
  "subquestions": [
    {
      "id": "SQ1",
      "question": "How many employees does Google have?",
      "depends_on": [],
      "reasoning": "Find Google's employee count"
    },
    {
      "id": "SQ2",
      "question": "How many employees does Microsoft have?",
      "depends_on": [],
      "reasoning": "Find Microsoft's employee count"
    },
    {
      "id": "SQ3",
      "question": "Which is larger: [SQ1_Answer] or [SQ2_Answer]?",
      "depends_on": ["SQ1", "SQ2"],
      "reasoning": "Compare the two employee counts"
    }
  ]
}

# Now decompose this question:

Question: __QUESTION__

Return ONLY valid JSON with the decomposition. Do not include any explanation outside the JSON.
"""
