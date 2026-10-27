BUILD_TRIPLET_PROMPT = '''You are an information extraction system.

Your task is to extract ALL factual information from the given passage and
represent it as a list of structured knowledge triplets.

Each triplet MUST follow the format:
(subject, relation, object)

INSTRUCTIONS:
1. Extract only facts explicitly stated or directly implied by the passage.
2. Do NOT hallucinate or add external knowledge.
3. Use concise, canonical entity names.
4. Normalize relations to be descriptive, verb-based, and lowercase
   (e.g., "located_in", "produced_by", "served_as", "elected_in").
5. Decompose complex facts into multiple atomic triplets.
6. Preserve temporal, locational, and numerical information as separate triplets.
7. If an entity has attributes (e.g., type, date, role), express them as triplets
   using relations like "has_type", "has_role", "has_date", "has_value".
8. When possible, prefer specific relations over generic ones like "related_to".
9. If the passage contains multiple entities, extract triplets for ALL of them.
10. If no factual triplets can be extracted, return an empty list.

IMPORTANT:
- Do not include explanations.
- Do not include duplicates.
- Output ONLY the triplets in the specified JSON format.

OUTPUT FORMAT:
[
  ("subject1", "relation1", "object1"),
  ("subject2", "relation2", "object2"),
  ...
]

Passage: """{{passage}}"""

Extracted Triplets:'''
