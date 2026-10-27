BUILD_HYPEREDGE_PROMPT = """

### Role
You are an advanced Knowledge Engineer specializing in Hypergraph construction. Your task is to extract complex relations from the text in the form of HyperEdges.

### Definition: HyperEdge
Unlike a standard Triplet (S-P-O), a HyperEdge can connect multiple entities and include contextual attributes within a single relationship.
- Format: { "relation": "...", "participants": { "role1": "entity1", "role2": "entity2", ... }, "context": { "time": "...", "location": "...", "attribute": "..." } }

### Extraction Guidelines
1. **Identify Complex Events**: Look for actions involving more than two participants (e.g., "A sold B to C for price D").
2. **Capture N-ary Relations**: Do not break a single event into multiple binary triplets if it loses context. Group them into one HyperEdge.
3. **Contextual Enrichment**: Attach metadata such as time, location, or manner of action directly to the edge.
4. **Entity Resolution**: Ensure all entities are normalized (replace pronouns with actual names).

### Output Format (JSON)
Return a list of HyperEdges in the following structure:
[
  {
    "edge_id": 1,
    "relation_type": "Action/Event Name",
    "participants": {
      "actor": "Entity A",
      "object": "Entity B",
      "recipient": "Entity C"
    },
    "attributes": {
      "temporal": "Time info",
      "spatial": "Location info",
      "modality": "How it happened"
    }
  }
]

---
### Input Text
{{corpus}}

---
### Final Response
Provide only the JSON array of HyperEdges.
"""
