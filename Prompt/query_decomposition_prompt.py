"""
Query Decomposition Prompt
===========================
Type-agnostic multi-hop question decomposition.
Supports 2-4 hop reasoning chains across diverse question patterns.

NO reliance on gold decompositions or question type labels.
Pure LLM-based reasoning decomposition.
"""

QUERY_DECOMPOSITION_PROMPT = """You are an expert at analyzing complex multi-hop questions and breaking them into logical reasoning steps.

Your task: Decompose the given question into a sequence of simpler sub-questions that can be answered through step-by-step retrieval and reasoning.

# Core Principles

1. **Identify Reasoning Chain**: What information is needed at each step?
2. **Atomic Sub-Questions**: Each sub-question should retrieve ONE specific piece of information
3. **Clear Dependencies**: Mark which sub-questions depend on previous answers
4. **Flexible Patterns**: Don't force predefined types - let the reasoning flow naturally

# Placeholder Syntax

Use **[SQ{N}_Answer]** to reference previous sub-question answers:
- Example: "Who directed [SQ1_Answer]?"
- Example: "Where was [SQ2_Answer] born?"

# Common Reasoning Patterns

## Pattern 1: Property Chain (2-3 hops)
"Where was the director of film X born?"
→ SQ1: "Who directed film X?"
→ SQ2: "Where was [SQ1_Answer] born?"

## Pattern 2: Parallel Comparison (2-3 hops)
"Which came first, A or B?"
→ SQ1: "When was A released?"
→ SQ2: "When was B released?"
→ SQ3: "Which is earlier: [SQ1_Answer] or [SQ2_Answer]?"

## Pattern 3: Multi-Entity Reasoning (3-4 hops)
"What is the relationship between A's property and B's property?"
→ SQ1: "What is A's property P?"
→ SQ2: "What is B's property Q?"
→ SQ3: "What is the relationship between [SQ1_Answer] and [SQ2_Answer]?"

## Pattern 4: Nested Properties (3-4 hops)
"What is the symbol of the team from the headquarters of company X?"
→ SQ1: "Where is the headquarters of company X?"
→ SQ2: "What team is from [SQ1_Answer]?"
→ SQ3: "What is the symbol of [SQ2_Answer]?"

## Pattern 5: Relationship Inference (2-3 hops)
"Who is the father-in-law of person X?"
→ SQ1: "Who is the spouse of person X?"
→ SQ2: "Who is the father of [SQ1_Answer]?"

## Pattern 6: Compositional Bridge-Compare (4+ hops)
"Which film's director died earlier, A or B?"
→ SQ1: "Who directed film A?"
→ SQ2: "Who directed film B?"
→ SQ3: "When did [SQ1_Answer] die?"
→ SQ4: "When did [SQ2_Answer] die?"
→ SQ5: "Which is earlier: [SQ3_Answer] or [SQ4_Answer]?"

# Guidelines

**Dependencies:**
- Independent sub-questions: `"depends_on": []`
- Sequential: `"depends_on": ["SQ1"]` or `"depends_on": ["SQ2"]`
- Multiple deps: `"depends_on": ["SQ1", "SQ2"]` (when synthesizing)

**Question Types** (for reference, not constraint):
- "compositional": Property chains
- "comparison": Parallel retrieval + compare
- "inference": Relationship reasoning
- "bridge": Sequential entity discovery
- "mixed": Combination of above

**Flexibility:**
- Support 2-4 hop questions (or more if needed)
- Don't force into rigid categories
- Let the reasoning chain emerge naturally
- Each sub-question should be simple enough for single-source retrieval

# Output Format

Return ONLY valid JSON:
```json
{
  "question_type": "<descriptive type>",
  "reasoning": "<brief explanation of decomposition strategy>",
  "subquestions": [
    {
      "id": "SQ1",
      "question": "<simple, atomic sub-question>",
      "depends_on": [],
      "reasoning": "<why this sub-question is needed>"
    },
    {
      "id": "SQ2",
      "question": "<may use [SQ1_Answer] placeholder>",
      "depends_on": ["SQ1"],
      "reasoning": "<why this sub-question is needed>"
    }
  ]
}
```

# Examples

## Example 1: Property Chain
Question: "Where was the director of film Doctor Krishna born?"

{
  "question_type": "compositional",
  "reasoning": "Need to first identify the director, then find their birthplace.",
  "subquestions": [
    {
      "id": "SQ1",
      "question": "Who is the director of film Doctor Krishna?",
      "depends_on": [],
      "reasoning": "First identify the director of the film"
    },
    {
      "id": "SQ2",
      "question": "Where was [SQ1_Answer] born?",
      "depends_on": ["SQ1"],
      "reasoning": "Find the birthplace of the identified director"
    }
  ]
}

## Example 2: Nested Properties
Question: "What body of water is by the headquarters location of Wipac?"

{
  "question_type": "compositional",
  "reasoning": "Need to find headquarters location first, then the nearby water body.",
  "subquestions": [
    {
      "id": "SQ1",
      "question": "Where is the headquarters of Wipac?",
      "depends_on": [],
      "reasoning": "Find the headquarters location"
    },
    {
      "id": "SQ2",
      "question": "What body of water is by [SQ1_Answer]?",
      "depends_on": ["SQ1"],
      "reasoning": "Find the water body near the headquarters location"
    }
  ]
}

## Example 3: Multi-Hop with Multiple Entities
Question: "What is the symbol of the Saints from the city where the headquarters of the manufacturer of McAfee's Benchmark called?"

{
  "question_type": "compositional",
  "reasoning": "Three-step chain: manufacturer → headquarters city → team symbol.",
  "subquestions": [
    {
      "id": "SQ1",
      "question": "Who is the manufacturer of McAfee's Benchmark?",
      "depends_on": [],
      "reasoning": "Identify the manufacturer"
    },
    {
      "id": "SQ2",
      "question": "Where is the headquarters of [SQ1_Answer]?",
      "depends_on": ["SQ1"],
      "reasoning": "Find the headquarters city"
    },
    {
      "id": "SQ3",
      "question": "What is the symbol of the Saints from [SQ2_Answer]?",
      "depends_on": ["SQ2"],
      "reasoning": "Find the team symbol for that city's Saints"
    }
  ]
}

## Example 4: Bridge-Comparison
Question: "Which film has the director who died earlier, A Doctor's Diary or Wild Rovers?"

{
  "question_type": "bridge_comparison",
  "reasoning": "Need to find both directors, their death dates, then compare.",
  "subquestions": [
    {
      "id": "SQ1",
      "question": "Who is the director of A Doctor's Diary?",
      "depends_on": [],
      "reasoning": "Identify first film's director"
    },
    {
      "id": "SQ2",
      "question": "Who is the director of Wild Rovers?",
      "depends_on": [],
      "reasoning": "Identify second film's director"
    },
    {
      "id": "SQ3",
      "question": "When did [SQ1_Answer] die?",
      "depends_on": ["SQ1"],
      "reasoning": "Find first director's death date"
    },
    {
      "id": "SQ4",
      "question": "When did [SQ2_Answer] die?",
      "depends_on": ["SQ2"],
      "reasoning": "Find second director's death date"
    },
    {
      "id": "SQ5",
      "question": "Which is earlier: [SQ3_Answer] or [SQ4_Answer]?",
      "depends_on": ["SQ3", "SQ4"],
      "reasoning": "Compare the two death dates to determine which director died earlier"
    }
  ]
}

## Example 5: Relationship Inference
Question: "Who is the father-in-law of Elizabeth Somerset, Baroness Herbert?"

{
  "question_type": "inference",
  "reasoning": "Need to find her spouse first, then the spouse's father.",
  "subquestions": [
    {
      "id": "SQ1",
      "question": "Who is the husband of Elizabeth Somerset, Baroness Herbert?",
      "depends_on": [],
      "reasoning": "Identify her spouse"
    },
    {
      "id": "SQ2",
      "question": "Who is the father of [SQ1_Answer]?",
      "depends_on": ["SQ1"],
      "reasoning": "Find the father of the identified spouse, which is the father-in-law"
    }
  ]
}

## Example 6: Simple Comparison
Question: "Are both Stephen R. Donaldson and Michael Moorcock science fiction writers?"

{
  "question_type": "comparison",
  "reasoning": "Check each person independently, then verify both.",
  "subquestions": [
    {
      "id": "SQ1",
      "question": "Is Stephen R. Donaldson a science fiction writer?",
      "depends_on": [],
      "reasoning": "Check first author"
    },
    {
      "id": "SQ2",
      "question": "Is Michael Moorcock a science fiction writer?",
      "depends_on": [],
      "reasoning": "Check second author"
    },
    {
      "id": "SQ3",
      "question": "Are both [SQ1_Answer] and [SQ2_Answer] true?",
      "depends_on": ["SQ1", "SQ2"],
      "reasoning": "Verify that BOTH are science fiction writers"
    }
  ]
}

# Now decompose this question

Question: __QUESTION__

**Important:**
- Return ONLY valid JSON
- No explanation text outside the JSON
- Use [SQ{N}_Answer] placeholders consistently
- Make each sub-question atomic and retrievable
- Support 2-4 hops (or more if needed)
- Let the reasoning chain emerge naturally from the question structure

JSON Response:
"""
