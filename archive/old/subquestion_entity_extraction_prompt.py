"""
Sub-Question Entity Extraction Prompt
=======================================
Extracts entities from sub-questions in a multi-hop QA pipeline.

Key differences from main query extraction:
1. Sub-questions are simpler and more focused
2. May have context from previous sub-question answers
3. Entity names may be substituted (e.g., "Taquini Plan" instead of "plan")
4. Focus on direct search targets, less ambiguity
"""

from Prompt.type_schema import ENTITY_TYPE_SCHEMA

# Build the prompt dynamically
_SUBQUESTION_PROMPT_TEMPLATE = """
You are an expert entity extraction system for sub-questions in a multi-hop question answering pipeline.

Given a sub-question (which may reference previous answers), extract ALL entities for retrieval.

**Context**: This is a sub-question that was decomposed from a larger multi-hop question. Previous sub-questions may have been answered already.

---

## ENTITY TYPE SCHEMA

{type_schema}

---

## OUTPUT FORMAT

Return a JSON object with entities classified by role and importance.
**IMPORTANT**: For each entity, provide ALL possible type/subtype combinations that might match in the database.

{{
  "entities": [
    {{
      "entity_name": <string>,          // Exact text from sub-question
      "possible_types": [                // List of all possible type/subtype pairs
        {{
          "type": <EntityType>,
          "subtype": <EntitySubtype>
        }},
        {{
          "type": <EntityType2>,
          "subtype": <EntitySubtype2>
        }},
        // ... more alternatives if applicable
      ],
      "role": <"target"|"attribute"|"context">,
      "importance": <"critical"|"important"|"optional">
    }},
    ...
  ]
}}

**Type Selection Guidelines:**
- Provide 2-3 most likely type/subtype combinations
- Order by likelihood (most likely first)
- For ambiguous entities, include all reasonable interpretations

---

## SUB-QUESTION CHARACTERISTICS

Sub-questions are typically:
1. **Simple and Direct**: Already decomposed, asking for ONE piece of information
2. **Focused**: Clear search target, less ambiguity than main query
3. **Short**: Usually 5-15 words
4. **Specific**: May contain concrete entity names from previous answers

---

## ROLE DEFINITIONS

1. **"target"**: Main entity the sub-question is asking about
   - The primary search target
   - Examples: "Watauga River", "Taquini Plan", "Stephen Graham"

2. **"attribute"**: Properties or characteristics being queried
   - Examples: "length", "proposer", "director", "free"

3. **"context"**: Location, time, or scope constraints
   - Examples: "Argentina", "2006", "northeast Tennessee"

---

## IMPORTANCE LEVELS

1. **"critical"**: Must be searched (main subject of sub-question)
2. **"important"**: Should be searched for better results
3. **"optional"**: Additional context, may help refine results

---

## EXTRACTION RULES FOR SUB-QUESTIONS

1. **Direct extraction**: Extract the main subject(s) of the sub-question
2. **Concrete entities**: Sub-questions often contain specific, already-identified entities
3. **No answer types**: Do NOT extract generic answer type words (who, what, which)
4. **Context matters**: If previous context is provided, focus on NEW entities in current sub-question
5. **Multiple types**: Provide 2-3 possible type/subtype pairs per entity, ordered by likelihood

---

## EXAMPLES

### Example 1: Simple Lookup (After Substitution)
**Previous Context:**
SQ1: "Who was the 23rd overall pick of the 2015 NHL Entry Draft?"
Answer: "Joel Eriksson Ek"

**Current Sub-Question:**
"In which international tournament did Joel Eriksson Ek help the United States national junior team win a bronze medal?"

**Output:**
{{
  "entities": [
    {{
      "entity_name": "Joel Eriksson Ek",
      "possible_types": [
        {{"type": "Person", "subtype": "Athlete"}},
        {{"type": "Person", "subtype": "HockeyPlayer"}},
        {{"type": "Person", "subtype": "SportsPlayer"}}
      ],
      "role": "target",
      "importance": "critical"
    }},
    {{
      "entity_name": "United States",
      "possible_types": [
        {{"type": "Location", "subtype": "Country"}},
        {{"type": "Location", "subtype": "GeopoliticalEntity"}}
      ],
      "role": "context",
      "importance": "important"
    }},
    {{
      "entity_name": "bronze medal",
      "possible_types": [
        {{"type": "Concept", "subtype": "Award"}},
        {{"type": "Concept", "subtype": "Achievement"}},
        {{"type": "Product", "subtype": "Trophy"}}
      ],
      "role": "attribute",
      "importance": "important"
    }},
    {{
      "entity_name": "junior team",
      "possible_types": [
        {{"type": "Concept", "subtype": "SportsTeam"}},
        {{"type": "Organization", "subtype": "Team"}},
        {{"type": "Concept", "subtype": "AgeGroup"}}
      ],
      "role": "attribute",
      "importance": "optional"
    }}
  ]
}}

---

### Example 2: Property Query
**Previous Context:**
SQ1: "Which river does the Bee Cliff overlook?"
Answer: "Watauga River"

**Current Sub-Question:**
"How long is Watauga River?"

**Output:**
{{
  "entities": [
    {{
      "entity_name": "Watauga River",
      "possible_types": [
        {{"type": "Location", "subtype": "River"}},
        {{"type": "Location", "subtype": "NaturalPlace"}},
        {{"type": "Location", "subtype": "GeographicFeature"}}
      ],
      "role": "target",
      "importance": "critical"
    }}
  ]
}}

---

### Example 3: Who Question
**Previous Context:**
SQ1: "What is the plan for free education in Argentina?"
Answer: "Taquini Plan"

**Current Sub-Question:**
"Who proposed Taquini Plan?"

**Output:**
{{
  "entities": [
    {{
      "entity_name": "Taquini Plan",
      "possible_types": [
        {{"type": "Concept", "subtype": "Policy"}},
        {{"type": "Concept", "subtype": "Program"}},
        {{"type": "Concept", "subtype": "EducationalSystem"}}
      ],
      "role": "target",
      "importance": "critical"
    }}
  ]
}}

---

### Example 4: Comparison Sub-Question
**Previous Context:**
SQ1: "Is Stephen R. Donaldson a science fiction writer?"
Answer: "Yes"
SQ2: "Is Michael Moorcock a science fiction writer?"
Answer: "Yes"

**Current Sub-Question:**
"Are both Yes and Yes true (i.e., are they both science fiction writers)?"

**Output:**
{{
  "entities": [
    {{
      "entity_name": "science fiction",
      "possible_types": [
        {{"type": "Concept", "subtype": "Genre"}},
        {{"type": "Concept", "subtype": "LiteraryGenre"}},
        {{"type": "Concept", "subtype": "Category"}}
      ],
      "role": "attribute",
      "importance": "important"
    }}
  ]
}}

---

### Example 5: First Sub-Question (No Previous Context)
**Previous Context:**
(None - this is the first sub-question)

**Current Sub-Question:**
"What is the plan for free education in Argentina?"

**Output:**
{{
  "entities": [
    {{
      "entity_name": "education",
      "possible_types": [
        {{"type": "Concept", "subtype": "SocialSystem"}},
        {{"type": "Concept", "subtype": "EducationalSystem"}},
        {{"type": "Concept", "subtype": "AcademicField"}}
      ],
      "role": "target",
      "importance": "critical"
    }},
    {{
      "entity_name": "Argentina",
      "possible_types": [
        {{"type": "Location", "subtype": "Country"}},
        {{"type": "Location", "subtype": "GeopoliticalEntity"}}
      ],
      "role": "context",
      "importance": "important"
    }},
    {{
      "entity_name": "free",
      "possible_types": [
        {{"type": "Concept", "subtype": "Attribute"}},
        {{"type": "Concept", "subtype": "Policy"}}
      ],
      "role": "attribute",
      "importance": "important"
    }}
  ]
}}

---

### Example 6: Location Query with Substitution
**Previous Context:**
SQ1: "In which international tournament did Joel Eriksson Ek help the United States win bronze?"
Answer: "2014 World Junior Ice Hockey Championships"

**Current Sub-Question:**
"In what city was 2014 World Junior Ice Hockey Championships held?"

**Output:**
{{
  "entities": [
    {{
      "entity_name": "2014 World Junior Ice Hockey Championships",
      "possible_types": [
        {{"type": "Event", "subtype": "SportsTournament"}},
        {{"type": "Event", "subtype": "Competition"}},
        {{"type": "Concept", "subtype": "SportsEvent"}}
      ],
      "role": "target",
      "importance": "critical"
    }}
  ]
}}

---

## CONTEXT HANDLING

**If previous context is provided:**
{{previous_context}}

**Focus on**: New entities in the current sub-question that need to be searched.
**Note**: 
- Entities from previous answers are typically already substituted into the sub-question
- Previous context may include retrieved passages from earlier sub-questions
- You can use information from previous passages to understand entity context, but still extract entities from the current sub-question

---

Now extract ALL entities from the following sub-question:

**Sub-Question:** __SUBQUESTION__

**Remember:**
- This is a sub-question (simpler and more focused than main query)
- Extract concrete entities for retrieval
- Do NOT extract answer type words (who, what, which, when, where)
- Provide 2-3 possible type/subtype pairs for each entity
- Consider the context of the multi-hop pipeline

Return only the JSON object, no additional text.
"""

SUBQUESTION_ENTITY_EXTRACTION_PROMPT = _SUBQUESTION_PROMPT_TEMPLATE.format(
    type_schema=ENTITY_TYPE_SCHEMA
)

