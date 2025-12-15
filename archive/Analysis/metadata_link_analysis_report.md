# Metadata Link Analysis for INCORRECT Questions

## Overview
We analyzed the metadata connections for 110 INCORRECT questions from the MuSiQue sample (200) evaluation. The goal was to understand how "Supporting Facts" (the documents we *should* have retrieved) are connected to each other versus how they are connected to "Distractors" (irrelevant documents).

## Key Statistics
- **Total INCORRECT Questions:** 110
- **Total Supporting Facts:** 292
- **Supporting Facts connected to OTHER Supporting Facts:** 163 (55.82%)
- **Supporting Facts connected to Distractors:** 201 (68.84%)
- **Isolated Supporting Facts:** 56 (19.18%)

## Link Pattern Analysis

### Supporting-Supporting (S-S) Links (Good Links)
These are the links we want to preserve.
- **Top Keys:**
    1. `relations.relation` (30) - *Weak signal (e.g., both have a "located_in" relation)*
    2. `relations.target` (24) - *Strong signal (e.g., both mention "Serbia")*
    3. `attributes.country` (4)
- **Top Values:**
    1. "located_in" (10)
    2. "American" (8)
    3. "Serbia" (6)
    4. "borders_with" (6)
    5. "Yongle Emperor" (4)

### Supporting-Distractor (S-D) Links (Bad Links/Noise)
These are the links that confuse the retriever.
- **Top Keys:**
    1. `relations.relation` (339) - *Huge source of noise*
    2. `relations.target` (48)
    3. `attributes.nationality` (21)
- **Top Values:**
    1. "located_in" (87)
    2. "American" (61)
    3. "United States" (33)
    4. "2" (25)
    5. "part_of" (23)

## Conclusion & Recommendation
The current graph construction is noisy because it treats generic relation types (like "located_in") and common attributes (like "American") as valid links.

**Recommendation:**
1.  **Blacklist Keys:** Ignore `relations.relation` when building the graph. Knowing two documents both have a "located_in" field doesn't mean they are related.
2.  **Stoplist Values:** Ignore common values like "American", "United States", and small integers ("1", "2", "3").
3.  **Prioritize Specific Keys:** Give higher weight to `relations.target`, `relations.entity`, and specific attribute names.
