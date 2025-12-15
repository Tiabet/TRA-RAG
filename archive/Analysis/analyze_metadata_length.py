"""
Analyze metadata length for Leonard Logsdail
"""
import json

# Leonard Logsdail metadata from DB
metadata = {
  "title": "Leonard Logsdail",
  "type": "Person",
  "subtype": "Businessperson",
  "attributes": {
    "full_name": "Leonard Logsdail",
    "birth_date": "September 11, 1950",
    "birth_place": {
      "city": "London",
      "country": "England",
      "type": "Location",
      "subtype": "City"
    },
    "profession": "bespoke tailor",
    "location": {
      "city": "Manhattan",
      "state": "New York",
      "country": "United States",
      "type": "Location",
      "subtype": "City"
    },
    "specialization": "men's suits",
    "description": "Described as one of the finest bespoke tailors in the men's suit business.",
    "crafting_method": "All of Logsdail's suits are crafted and perfected on-site in his New York City location.",
    "contribution": "Credited for making Savile Row tailoring a local option in New York.",
    "notable_works": [
      {
        "type": "high-end suits",
        "details": "Including lining jackets with Hermes silk scarves."
      },
      {
        "type": "film collaborations",
        "details": "Created suits for award-winning films."
      }
    ],
    "recognition": "Recognized as one of cinema's most sought-after tailors."
  },
  "relations": [
    {
      "relation": "has_suits_made_for",
      "target": {
        "name": "Larry Kudlow",
        "type": "Person",
        "subtype": "Businessperson",
        "description": "CNBC talk show host, economist and fashion icon."
      }
    },
    {
      "relation": "collaborated_with",
      "target": [
        {
          "name": "Steven Spielberg",
          "type": "Person",
          "subtype": "Director"
        },
        {
          "name": "Robert de Niro",
          "type": "Person",
          "subtype": "Actor"
        },
        {
          "name": "Oliver Stone",
          "type": "Person",
          "subtype": "Director"
        },
        {
          "name": "Ridley Scott",
          "type": "Person",
          "subtype": "Director"
        },
        {
          "name": "Martin Scorcese",
          "type": "Person",
          "subtype": "Director"
        }
      ]
    },
    {
      "relation": "appeared_in",
      "target": [
        {
          "title": "The Wolf of Wall Street",
          "type": "WorkOfArt",
          "subtype": "Film"
        },
        {
          "title": "The Good Shepherd",
          "type": "WorkOfArt",
          "subtype": "Film"
        }
      ],
      "details": "Had a cameo acting role as a tailor."
    }
  ]
}

# Analyze lengths
attrs_str = json.dumps(metadata['attributes'])
rels_str = json.dumps(metadata['relations'])
total_str = json.dumps(metadata)

attrs_len = len(attrs_str)
rels_len = len(rels_str)
total_len = len(total_str)

print("="*80)
print("Metadata Length Analysis for Leonard Logsdail")
print("="*80)
print(f"\nTotal metadata length: {total_len} chars")
print(f"  - Attributes: {attrs_len} chars ({attrs_len/total_len*100:.1f}%)")
print(f"  - Relations: {rels_len} chars ({rels_len/total_len*100:.1f}%)")
print(f"\nEstimated tokens (char/4): ~{total_len//4} tokens")

print("\n" + "="*80)
print("Old Truncation Limits:")
print("="*80)
print(f"Stage 1-A (value[:100]):")
print(f"  - Attributes truncated: {attrs_str[:100]}...")
print(f"  - Relations truncated: {rels_str[:100]}...")

print(f"\nStage 1-B (value[:150]):")
print(f"  - Attributes truncated: {attrs_str[:150]}...")
print(f"  - Relations truncated: {rels_str[:150]}...")

print("\n" + "="*80)
print("Problem Analysis:")
print("="*80)
print(f"With 100-char limit: Relations field completely lost (needs {rels_len} chars)")
print(f"With 150-char limit: Only partial relations shown, 'appeared_in' at position {rels_str.find('appeared_in')}")
print(f"Without truncation: Full {total_len} chars sent to LLM per candidate")

print("\n" + "="*80)
print("Recommendations:")
print("="*80)
print("1. Smart truncation: Keep important fields (relations) intact, truncate verbose fields (attributes)")
print("2. Summarize long fields: Extract key info from nested structures")
print("3. Prioritize by relevance: Show relations first, then critical attributes")
print("4. Set per-field limits: Different truncation for different field types")
