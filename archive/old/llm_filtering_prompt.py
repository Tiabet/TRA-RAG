"""
LLM-based Filtering Prompt
===========================
2차 필터링: Type/subtype으로 1차 필터링된 후보들을 LLM으로 최종 필터링
"""

LLM_FILTERING_PROMPT = """
You are an expert entity matching system for a knowledge base retrieval task.

Given:
1. A **query** asking for information
2. An **extracted entity** from the query (with type/subtype)
3. A list of **candidate passages** that were retrieved based on type/subtype matching

Your task is to identify which candidate passages are **truly relevant** to answering the query.

---

### MATCHING CRITERIA

A passage is relevant if:
✓ It contains information about the entity mentioned in the query
✓ It describes the same real-world entity (accounting for spelling variations, aliases, abbreviations)
✓ The passage could help answer the question
✓ **IMPORTANT**: Even if type/subtype differs, if the passage REFERENCES or MENTIONS the target entity in its metadata (e.g., in title_reference, description, attributes), it should be considered relevant

A passage is NOT relevant if:
✗ It mentions a different entity with a similar name (and no connection to the query entity)
✗ It's about a completely unrelated topic with no mention of the query entity
✗ The keyword overlap is purely coincidental

**Special Cases to INCLUDE:**
- Films/books/works that reference the entity (even if entity type is different)
- Passages where the entity appears in nested attributes, not just title
- Passages that define, explain, or reference the entity contextually

---

### SPELLING VARIATIONS & ALIASES

Consider these as MATCHING:
- "NHL" vs "National Hockey League"
- "2015 NHL Entry Draft" vs metadata with separate "2015", "NHL", "Draft" attributes
- "Charles de Gaulle Airport" vs "Roissy Airport"
- "Gweilo" vs "ghost man" (Chinese slang)
- "UK" vs "United Kingdom"
- Minor spelling differences (e.g., "Theater" vs "Theatre")

---

### INPUT FORMAT

Query: <the original question>
Extracted Entity: 
  - Name: <entity_name>
  - Type: <entity_type>
  - Subtype: <entity_subtype>

Candidate Passages:
[
  {
    "title": <passage_title>,
    "type": <passage_type>,
    "subtype": <passage_subtype>,
    "snippet": <key attributes from metadata>
  },
  ...
]

---

### OUTPUT FORMAT

Return a JSON object:

{
  "relevant_passages": [
    {
      "title": <passage_title>,
      "confidence": <"high" | "medium" | "low">,
      "reasoning": <brief explanation why it matches>
    }
  ],
  "filtered_out": [
    {
      "title": <passage_title>,
      "reasoning": <brief explanation why it doesn't match>
    }
  ]
}

---

### EXAMPLES

#### Example 1: Spelling Variation Match

**Input:**
Query: "The Roissy Airport connects to Paris and cities in what countries?"
Extracted Entity:
  - Name: Roissy Airport
  - Type: Location
  - Subtype: Airport

Candidate Passages:
[
  {
    "title": "Charles de Gaulle Airport",
    "type": "Location",
    "subtype": "Airport",
    "snippet": "Also known as Roissy Airport, located in Roissy-en-France"
  },
  {
    "title": "Paris Orly Airport",
    "type": "Location",
    "subtype": "Airport",
    "snippet": "Second-busiest airport serving Paris"
  }
]

**Output:**
{
  "relevant_passages": [
    {
      "title": "Charles de Gaulle Airport",
      "confidence": "high",
      "reasoning": "Explicitly states 'Also known as Roissy Airport' - this is an alias match"
    }
  ],
  "filtered_out": [
    {
      "title": "Paris Orly Airport",
      "reasoning": "Different airport - no indication this is Roissy Airport"
    }
  ]
}

---

#### Example 2: Fragmented Entity Match

**Input:**
Query: "What team selected Connor McDavid in the 2015 NHL Entry Draft?"
Extracted Entity:
  - Name: 2015 NHL Entry Draft
  - Type: Event
  - Subtype: SportsEvent

Candidate Passages:
[
  {
    "title": "Connor McDavid",
    "type": "Person",
    "subtype": "Athlete",
    "snippet": "Selected first overall by Edmonton Oilers in 2015, NHL Entry Draft"
  },
  {
    "title": "2015 NHL Entry Draft",
    "type": "Event",
    "subtype": "SportsEvent",
    "snippet": "Annual NHL draft held June 26-27, 2015"
  },
  {
    "title": "2014 NHL Entry Draft",
    "type": "Event",
    "subtype": "SportsEvent",
    "snippet": "Previous year's draft"
  }
]

**Output:**
{
  "relevant_passages": [
    {
      "title": "Connor McDavid",
      "confidence": "high",
      "reasoning": "Contains '2015', 'NHL', 'Entry Draft' - mentions the exact event in context"
    },
    {
      "title": "2015 NHL Entry Draft",
      "confidence": "high",
      "reasoning": "Exact match - the event itself"
    }
  ],
  "filtered_out": [
    {
      "title": "2014 NHL Entry Draft",
      "reasoning": "Wrong year - 2014 not 2015"
    }
  ]
}

---

#### Example 3: Type Mismatch Filter

**Input:**
Query: "What Cantonese slang term means 'ghost man'?"
Extracted Entity:
  - Name: ghost man
  - Type: Concept
  - Subtype: SlangTerm

Candidate Passages:
[
  {
    "title": "Gweilo",
    "type": "Concept",
    "subtype": "SlangTerm",
    "snippet": "Cantonese slang for Westerners, literally 'ghost man'"
  },
  {
    "title": "Ghosts (2006 film)",
    "type": "WorkOfArt",
    "subtype": "Film",
    "snippet": "Title references Cantonese 'Gweilo' meaning 'ghost man'"
  }
]

**Output:**
{
  "relevant_passages": [
    {
      "title": "Gweilo",
      "confidence": "high",
      "reasoning": "Exact match - Cantonese slang term that literally means 'ghost man'"
    },
    {
      "title": "Ghosts (2006 film)",
      "confidence": "medium",
      "reasoning": "Film whose title references the Cantonese term 'Gweilo' meaning 'ghost man' - contains relevant context about the term"
    }
  ],
  "filtered_out": []
}

---

Now perform the filtering task:

Query: {{QUERY}}
Extracted Entity:
  - Name: {{ENTITY_NAME}}
  - Type: {{ENTITY_TYPE}}
  - Subtype: {{ENTITY_SUBTYPE}}

Candidate Passages:
{{CANDIDATES}}

Return only the JSON object, no additional text.
"""
