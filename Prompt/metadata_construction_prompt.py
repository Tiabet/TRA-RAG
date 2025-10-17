from Prompt.type_schema import ENTITY_TYPE_SCHEMA

# Construct the prompt with schema insertion
_prompt_template = """
You are an expert metadata extraction engine specialized in transforming natural language passages into fully structured JSON metadata objects.

You will be given a passage that describes one or more entities such as people, organizations, locations, works of art, events, products, biological entities, or concepts.  
Your task is to extract **ALL information without ANY omission** and convert it into a structured metadata JSON following the schema below.

---

### ⚠️ CRITICAL REQUIREMENTS - NO INFORMATION LOSS

- **MANDATORY**: The output **MUST include EVERY SINGLE factual detail** from the passage
- **NO omissions**: Every name, date, number, location, relationship, description, attribute MUST be captured
- **NO summarizations**: Use exact quotes and full details from the original text
- **NO interpretation**: Extract facts as stated, do not infer or add information not in the passage
- Metadata must be **hierarchical, semantically precise, and fully structured JSON**
- Each entity must have:
  - `"type"` and `"subtype"` fields based on the **EntityTypeSchema** below  
- All relationships must be explicitly captured under a `"relations"` key
- The final output must be **pure JSON**, with no commentary or explanations

### Information Preservation Checklist:
✓ All names (people, places, organizations, works)
✓ All dates (birth, death, founding, release, events)
✓ All numbers (quantities, measurements, statistics, rankings)
✓ All locations (countries, states, cities, addresses, coordinates)
✓ All descriptions and qualifiers
✓ All relationships and connections
✓ All attributes and characteristics
✓ All quoted text and terminology
✓ All context and background information

---

### JSON SCHEMA TEMPLATE

{
  "title": <string>,
  "type": <EntityType>,
  "subtype": <EntitySubtype>,
  "attributes": {
    <attribute_name>: <value_or_nested_entity>
  },
  "relations": [
    {
      "relation": <string>,
      "target": <entity_or_list_of_entities>
    }
  ]
}

---

### ENTITY TYPE SCHEMA

"""

metadata_construction_prompt = _prompt_template + ENTITY_TYPE_SCHEMA + """

---

### EXAMPLES

#### Example 1
**Input:**
[["Can't Touch It", ["\"Can't Touch It\" is a song by Australian singer and songwriter Ricki-Lee Coulter.", " It was written by Coulter with Brian Kierulf and Joshua M. Schwartz of KNS Productions, who also produced the song.", " \"Can't Touch It\" was released as the lead single from Coulter's second studio album \"Brand New Day\" on 4 August 2007.", " Upon its release, \"Can't Touch It\" peaked at number two on the ARIA Singles Chart and number one on the ARIA Dance Singles Chart, where it remained for eight consecutive weeks.", " It was certified platinum by the Australian Recording Industry Association for shipments of 70,000 copies."]]]

**Output:**
{
  "title": "Can't Touch It",
  "type": "WorkOfArt",
  "subtype": "Song",
  "attributes": {
    "artist": {
      "name": "Ricki-Lee Coulter",
      "type": "Person",
      "subtype": "Musician",
      "nationality": "Australian"
    },
    "writers": [
      {"name": "Ricki-Lee Coulter", "type": "Person", "subtype": "Musician"},
      {"name": "Brian Kierulf", "type": "Person", "subtype": "Musician"},
      {"name": "Joshua M. Schwartz", "type": "Person", "subtype": "Musician"}
    ],
    "producers": [
      {"name": "Brian Kierulf", "type": "Person", "subtype": "Musician"},
      {"name": "Joshua M. Schwartz", "type": "Person", "subtype": "Musician"},
      {"name": "KNS Productions", "type": "Organization", "subtype": "MediaOrganization"}
    ],
    "album": {
      "title": "Brand New Day",
      "type": "WorkOfArt",
      "subtype": "Album"
    },
    "release_date": "4 August 2007",
    "chart_performance": {
      "ARIA_Singles_Chart": {"peak_position": 2},
      "ARIA_Dance_Singles_Chart": {"peak_position": 1, "duration_at_peak": "8 consecutive weeks"}
    },
    "certification": {
      "organization": {
        "name": "Australian Recording Industry Association (ARIA)",
        "type": "Organization",
        "subtype": "NonProfitOrganization"
      },
      "level": "Platinum",
      "shipments": "70,000 copies"
    }
  },
  "relations": [
    {"relation": "included_in_album", "target": {"title": "Brand New Day", "type": "WorkOfArt", "subtype": "Album"}},
    {"relation": "produced_by", "target": {"name": "KNS Productions", "type": "Organization", "subtype": "MediaOrganization"}},
    {"relation": "written_by", "target": [
      {"name": "Ricki-Lee Coulter", "type": "Person", "subtype": "Musician"},
      {"name": "Brian Kierulf", "type": "Person", "subtype": "Musician"},
      {"name": "Joshua M. Schwartz", "type": "Person", "subtype": "Musician"}
    ]}
  ]
}

---

#### Example 2
**Input:**
["Tosside", ["Tosside is a small village on the border of North Yorkshire and Lancashire in Northern England.", " It lies within the Forest of Bowland, and is between the villages of Slaidburn in Lancashire and Wigglesworth in North Yorkshire.", " It lies 11.5 miles north of Clitheroe and 17 miles northwest of Skipton.", " The village is 870 ft above sea level and lies at 54.0001°N / 2.35436°W on the B6478."]]

**Output:**
{
  "title": "Tosside",
  "type": "Location",
  "subtype": "Village",
  "attributes": {
    "full_name": "Tosside",
    "description": "A small village on the border of North Yorkshire and Lancashire in Northern England.",
    "country": "England",
    "regions": [
      {"name": "North Yorkshire", "type": "Location", "subtype": "County"},
      {"name": "Lancashire", "type": "Location", "subtype": "County"}
    ],
    "area": {"name": "Forest of Bowland", "type": "Location", "subtype": "NaturalPlace"},
    "nearby_villages": [
      {"name": "Slaidburn", "type": "Location", "subtype": "Village", "county": "Lancashire"},
      {"name": "Wigglesworth", "type": "Location", "subtype": "Village", "county": "North Yorkshire"}
    ],
    "distances": {
      "Clitheroe": {"distance_miles": 11.5, "direction": "north"},
      "Skipton": {"distance_miles": 17, "direction": "northwest"}
    },
    "elevation_ft": 870,
    "coordinates": {"latitude": 54.0001, "longitude": -2.35436},
    "road": "B6478"
  },
  "relations": [
    {"relation": "located_within", "target": {"name": "Forest of Bowland", "type": "Location", "subtype": "NaturalPlace"}},
    {"relation": "border_between", "target": [
      {"name": "North Yorkshire", "type": "Location", "subtype": "County"},
      {"name": "Lancashire", "type": "Location", "subtype": "County"}
    ]},
    {"relation": "near", "target": [
      {"name": "Slaidburn", "type": "Location", "subtype": "Village"},
      {"name": "Wigglesworth", "type": "Location", "subtype": "Village"},
      {"name": "Clitheroe", "type": "Location", "subtype": "Town"},
      {"name": "Skipton", "type": "Location", "subtype": "Town"}
    ]}
  ]
}

#### Example 3
**Input:**
[["Richard Masur", ["Richard Masur (born November 20, 1948) is an American actor who has appeared in more than 80 movies.", " From 1995 to 1999, he served two terms as president of the Screen Actors Guild (SAG).", " Masur currently sits on the Corporate Board of the Motion Picture & Television Fund."]]

**Output:**
{
  "title": "Richard Masur",
  "type": "Person",
  "subtype": "Actor",
  "attributes": {
    "full_name": "Richard Masur",
    "birth_date": "November 20, 1948",
    "nationality": "American",
    "occupations": ["actor"],
    "filmography_count": "appeared in more than 80 movies",
    "positions": [
      {
        "title": "President",
        "organization": {
          "name": "Screen Actors Guild (SAG)",
          "type": "Organization",
          "subtype": "NonProfitOrganization"
        },
        "term": "1995–1999",
        "term_count": 2
      },
      {
        "title": "Corporate Board Member",
        "organization": {
          "name": "Motion Picture & Television Fund",
          "type": "Organization",
          "subtype": "NonProfitOrganization"
        },
        "status": "current"
      }
    ]
  },
  "relations": [
    {
      "relation": "served_as_president_of",
      "target": {
        "name": "Screen Actors Guild (SAG)",
        "type": "Organization",
        "subtype": "NonProfitOrganization"
      },
      "period": "1995–1999"
    },
    {
      "relation": "member_of_board",
      "target": {
        "name": "Motion Picture & Television Fund",
        "type": "Organization",
        "subtype": "NonProfitOrganization"
      },
      "status": "current"
    }
  ]
}

---

### 🧠 Instruction:
- Now, transform the following passage into structured JSON metadata using the schema above.
- Output **only** the final JSON object — no explanations, notes, or comments.
- The output **must include every factual detail** from the passage — no omissions or summarizations.  
- Metadata must be **hierarchical, semantically precise, and fully structured JSON**.  
- Each entity must have:
  - `"type"` and `"subtype"` fields based on the **EntityTypeSchema** below.  
- All relationships must be explicitly captured under a `"relations"` key.  
- The final output must be **pure JSON**, with no commentary or explanations.

### passage:
{{input}}
"""