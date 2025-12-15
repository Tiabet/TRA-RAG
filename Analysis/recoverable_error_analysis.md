# Recoverable Error Analysis

## Objective
Identify questions where the current retrieval pipeline failed (`INCORRECT`) but the "Upper Bound" (Gold Passages + Final Prompt) succeeded (`CORRECT`). These "Recoverable Errors" represent cases where the LLM is capable of answering if the retrieval provides the correct context.

## Methodology
1.  **Filter:** Intersect `INCORRECT` results from `llm_eval_test_musique_v4_200_results.json` with `CORRECT` results from `llm_eval_upper_bound_musique.json`.
2.  **Count:** Found **49** recoverable questions.
3.  **Analysis:** Analyze the metadata connectivity of the "Supporting Facts" for these 49 questions to understand why retrieval failed. Specifically, compare Supporting-Supporting (S-S) links vs. Supporting-Distractor (S-D) links.

## Findings: The "Generic Relation" Noise

The analysis reveals a critical flaw in the graph construction logic: **Generic relation names are creating massive noise.**

### Signal vs. Noise Ratio
| Link Type | S-S Count (Signal) | S-D Count (Noise) | Ratio (Noise/Signal) |
| :--- | :--- | :--- | :--- |
| `relations.relation` | 22 | **174** | **7.9x** |
| `relations.target` | 20 | 24 | 1.2x |

### The Problem
The system is creating links between documents simply because they share the same *type* of relation, not because they share the same *entity*.

*   **Example:** Document A has `relation: "located_in"` (target: Paris). Document B has `relation: "located_in"` (target: London).
*   **Current Behavior:** The system links A and B because they both share the value `"located_in"` under the key `relations.relation`.
*   **Result:** Every document with a location is connected to every other document with a location. This creates a "super-node" of generic terms that short-circuits the retrieval path, flooding the context with distractors.

### Top Noisy Values
The following values are responsible for the majority of false positive links:
1.  `located_in` (53 S-D links)
2.  `American` (30 S-D links)
3.  `United States` (22 S-D links)
4.  `part_of` (13 S-D links)
5.  `True` (12 S-D links)

## Recommendation
**Modify the Graph Construction Logic:**
1.  **Blacklist `relations.relation`:** Do not create links based on the *name* of the relation. Only link based on the `relations.target`.
2.  **Stoplist Generic Values:** Consider adding a stoplist for extremely common attribute values like "American", "United States", "True", "False", numbers ("2", "3").
3.  **Strict Typing:** Only link `relations.target` to `title` or other `relations.target`, rather than allowing cross-type matching with generic text fields if possible.

By implementing these filters, we expect to eliminate the 174+ noise links while preserving the 20+ valid `relations.target` links, significantly improving the retrieval precision for these 49 recoverable questions.
