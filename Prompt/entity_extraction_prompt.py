"""
Entity Extraction Prompt
=========================
Extracts the primary entity to start search from a given query.
"""

from Prompt.type_schema import ENTITY_TYPE_SCHEMA

# Build the prompt dynamically to avoid f-string issues with curly braces
_PROMPT_TEMPLATE = """
You are an expert entity extraction system for question answering tasks.

Given a question, your task is to identify the **primary entity** that should be searched first in a knowledge base to answer the question. This entity will serve as the starting point for knowledge graph traversal.

---

## ENTITY TYPE SCHEMA

{type_schema}

---

## OUTPUT FORMAT

Return a JSON object with the following structure:

**For single entity questions:**
{{
  "entities": [
    {{
      "entity_name": <string>,
      "type": <EntityType>,
      "subtype": <EntitySubtype>
    }}
  ]
}}

**For multiple entity questions (comparison, multiple subjects, etc.):**
{{
  "entities": [
    {{
      "entity_name": <string>,
      "type": <EntityType>,
      "subtype": <EntitySubtype>
    }},
    {{
      "entity_name": <string>,
      "type": <EntityType>,
      "subtype": <EntitySubtype>
    }}
  ]
}}

---

## EXTRACTION RULES

1. **Primary Entity**: Extract the most specific entity/entities that need to be looked up
2. **Type Assignment**: Assign the most appropriate type and subtype from the schema
3. **Canonical Form**: Use the exact name as it appears in the question
4. **Multiple Entities**: For comparison questions or when multiple entities are explicitly mentioned and need to be looked up, extract ALL relevant entities
5. **Single vs Multiple**: 
   - Single entity: Questions with one clear subject
   - Multiple entities: Comparison questions ("which A or B"), questions with multiple explicit subjects

---

## EXAMPLES

### Example 1: Geographic Location Query
**Input Question:**
"What mountain range does Mount Kilimanjaro belong to, and what is its highest peak?"

**Output:**
{{
  "entities": [
    {{
      "entity_name": "Mount Kilimanjaro",
      "type": "Location",
      "subtype": "NaturalPlace"
    }}
  ]
}}

---

### Example 2: Person-Organization Relationship
**Input Question:**
"Who was the founder of SpaceX, and in which year did they establish Tesla Motors?"

**Output:**
{{
  "entities": [
    {{
      "entity_name": "SpaceX",
      "type": "Organization",
      "subtype": "Company"
    }},
    {{
      "entity_name": "Tesla Motors",
      "type": "Organization",
      "subtype": "Company"
    }}
  ]
}}

---

### Example 3: Comparison Query
**Input Question:**
"Which novel was published earlier, To Kill a Mockingbird or The Catcher in the Rye?"

**Output:**
{{
  "entities": [
    {{
      "entity_name": "To Kill a Mockingbird",
      "type": "WorkOfArt",
      "subtype": "Book"
    }},
    {{
      "entity_name": "The Catcher in the Rye",
      "type": "WorkOfArt",
      "subtype": "Book"
    }}
  ]
}}

---

### Example 4: Multi-hop Historical Event
**Input Question:**
"In which city did the signing ceremony of the treaty that ended World War I take place?"

**Output:**
{{
  "entities": [
    {{
      "entity_name": "Treaty of Versailles",
      "type": "Event",
      "subtype": "HistoricalEvent"
    }}
  ]
}}

---

### Example 5: Compositional Artist Query
**Input Question:**
"What genre of music is primarily associated with the composer of The Four Seasons?"

**Output:**
{{
  "entities": [
    {{
      "entity_name": "The Four Seasons",
      "type": "WorkOfArt",
      "subtype": "Opera"
    }}
  ]
}}

---

### Example 6: Multiple Entity Inference
**Input Question:**
"What is the distance between the Statue of Liberty and the Golden Gate Bridge?"

**Output:**
{{
  "entities": [
    {{
      "entity_name": "Statue of Liberty",
      "type": "Location",
      "subtype": "Landmark"
    }},
    {{
      "entity_name": "Golden Gate Bridge",
      "type": "Location",
      "subtype": "Landmark"
    }}
  ]
}}

---

Now extract the primary entity from the following question:

**Question:** __QUESTION__

Return only the JSON object, no additional text.
"""

ENTITY_EXTRACTION_PROMPT = _PROMPT_TEMPLATE.format(type_schema=ENTITY_TYPE_SCHEMA)
