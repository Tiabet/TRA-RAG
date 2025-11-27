"""
Metadata Construction Prompt (Simplified - No Type/Subtype Schema)
==================================================================
Type/subtype 스키마 없이 자유로운 형식으로 메타데이터 추출
"""

metadata_construction_prompt = """
You are an expert metadata extraction engine specialized in transforming natural language passages into fully structured JSON metadata objects.

You will be given a passage that describes one or more entities such as people, organizations, locations, works of art, events, products, or concepts.  
Your task is to extract **ALL information without ANY omission** and convert it into a structured metadata JSON.

---

### ⚠️ CRITICAL REQUIREMENTS - NO INFORMATION LOSS

- **MANDATORY**: The output **MUST include EVERY SINGLE factual detail** from the passage
- **NO omissions**: Every name, date, number, location, relationship, description, attribute MUST be captured
- **NO summarizations**: Use exact quotes and full details from the original text
- **NO interpretation**: Extract facts as stated, do not infer or add information not in the passage
- Metadata must be **hierarchical, semantically precise, and fully structured JSON**
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
  "attributes": {
    <attribute_name>: <value_or_nested_object>
  },
  "relations": [
    {
      "relation": <string>,
      "target": <entity_name_or_object>
    }
  ]
}

---

### EXAMPLES

#### Example 1
**Input:**
[["Can't Touch It", ["\"Can't Touch It\" is a song by Australian singer and songwriter Ricki-Lee Coulter.", " It was written by Coulter with Brian Kierulf and Joshua M. Schwartz of KNS Productions, who also produced the song.", " \"Can't Touch It\" was released as the lead single from Coulter's second studio album \"Brand New Day\" on 4 August 2007.", " Upon its release, \"Can't Touch It\" peaked at number two on the ARIA Singles Chart and number one on the ARIA Dance Singles Chart, where it remained for eight consecutive weeks.", " It was certified platinum by the Australian Recording Industry Association for shipments of 70,000 copies."]]]

**Output:**
{
  "title": "Can't Touch It",
  "attributes": {
    "artist": {
      "name": "Ricki-Lee Coulter",
      "nationality": "Australian",
      "roles": ["singer", "songwriter"]
    },
    "writers": ["Ricki-Lee Coulter", "Brian Kierulf", "Joshua M. Schwartz"],
    "producers": ["Brian Kierulf", "Joshua M. Schwartz", "KNS Productions"],
    "album": "Brand New Day",
    "album_order": "second studio album",
    "single_type": "lead single",
    "release_date": "4 August 2007",
    "chart_performance": {
      "ARIA_Singles_Chart": {"peak_position": 2},
      "ARIA_Dance_Singles_Chart": {"peak_position": 1, "duration_at_peak": "8 consecutive weeks"}
    },
    "certification": {
      "organization": "Australian Recording Industry Association",
      "level": "Platinum",
      "shipments": "70,000 copies"
    }
  },
  "relations": [
    {"relation": "included_in_album", "target": "Brand New Day"},
    {"relation": "produced_by", "target": "KNS Productions"},
    {"relation": "written_by", "target": ["Ricki-Lee Coulter", "Brian Kierulf", "Joshua M. Schwartz"]}
  ]
}

---

#### Example 2
**Input:**
["Tosside", ["Tosside is a small village on the border of North Yorkshire and Lancashire in Northern England.", " It lies within the Forest of Bowland, and is between the villages of Slaidburn in Lancashire and Wigglesworth in North Yorkshire.", " It lies 11.5 miles north of Clitheroe and 17 miles northwest of Skipton.", " The village is 870 ft above sea level and lies at 54.0001°N / 2.35436°W on the B6478."]]

**Output:**
{
  "title": "Tosside",
  "attributes": {
    "description": "A small village on the border of North Yorkshire and Lancashire in Northern England",
    "country": "England",
    "region": "Northern England",
    "counties": ["North Yorkshire", "Lancashire"],
    "area": "Forest of Bowland",
    "nearby_villages": [
      {"name": "Slaidburn", "county": "Lancashire"},
      {"name": "Wigglesworth", "county": "North Yorkshire"}
    ],
    "distances": {
      "from_Clitheroe": {"miles": 11.5, "direction": "north"},
      "from_Skipton": {"miles": 17, "direction": "northwest"}
    },
    "elevation_ft": 870,
    "coordinates": {"latitude": 54.0001, "longitude": -2.35436},
    "road": "B6478"
  },
  "relations": [
    {"relation": "located_within", "target": "Forest of Bowland"},
    {"relation": "border_between", "target": ["North Yorkshire", "Lancashire"]},
    {"relation": "near", "target": ["Slaidburn", "Wigglesworth", "Clitheroe", "Skipton"]}
  ]
}

#### Example 3
**Input:**
[["Richard Masur", ["Richard Masur (born November 20, 1948) is an American actor who has appeared in more than 80 movies.", " From 1995 to 1999, he served two terms as president of the Screen Actors Guild (SAG).", " Masur currently sits on the Corporate Board of the Motion Picture & Television Fund."]]

**Output:**
{
  "title": "Richard Masur",
  "attributes": {
    "full_name": "Richard Masur",
    "birth_date": "November 20, 1948",
    "nationality": "American",
    "occupation": "actor",
    "filmography_count": "more than 80 movies",
    "positions": [
      {
        "title": "President",
        "organization": "Screen Actors Guild (SAG)",
        "term": "1995–1999",
        "term_count": "two terms"
      },
      {
        "title": "Corporate Board Member",
        "organization": "Motion Picture & Television Fund",
        "status": "current"
      }
    ]
  },
  "relations": [
    {"relation": "served_as_president_of", "target": "Screen Actors Guild (SAG)", "period": "1995–1999"},
    {"relation": "member_of_board", "target": "Motion Picture & Television Fund"}
  ]
}

---

### 🧠 Instruction:
- Now, transform the following passage into structured JSON metadata using the schema above.
- Output **only** the final JSON object — no explanations, notes, or comments.
- The output **must include every factual detail** from the passage — no omissions or summarizations.  
- Metadata must be **hierarchical, semantically precise, and fully structured JSON**.  
- All relationships must be explicitly captured under a `"relations"` key.  
- The final output must be **pure JSON**, with no commentary or explanations.

### passage:
{{input}}
"""