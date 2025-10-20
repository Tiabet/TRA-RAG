"""
Hybrid Entity Retrieval System
================================
Combines value-based matching and type-based LLM filtering.

Pipeline:
  Stage 1-A: Value-based entity matching (FTS on all metadata values)
  Stage 1-B: Type/Subtype filtering → LLM title filtering
  Stage 2: Merge results from both stages
"""
import asyncio
import os
import json
from typing import List, Dict, Optional, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv

from metadata_db import MetadataDB
from Prompt.entity_extraction_prompt import ENTITY_EXTRACTION_PROMPT

# Load environment variables
load_dotenv()


def initialize_llm_client():
    """Initialize the AsyncOpenAI client with Alice API settings"""
    api_key = os.getenv('ALICE_OPENAI_KEY')
    base_url = os.getenv('ALICE_CHAT_URL')
    
    if not api_key or not base_url:
        raise ValueError("ALICE_OPENAI_KEY and ALICE_CHAT_URL must be set in .env file")
    
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )


# LLM Title Filtering Prompt
TITLE_FILTERING_PROMPT = """You are an expert at identifying relevant passages for answering questions.

Given:
- A QUERY
- An ENTITY extracted from the query (with type/subtype)
- A list of CANDIDATE TITLES (from passages with matching type/subtype)

Your task: Filter the candidate titles to keep only those that are RELEVANT to answering the query about the entity.

FILTERING CRITERIA:
1. **Direct match**: Title is about the entity or directly related concept
2. **Contextual relevance**: Title provides information needed to answer the query
3. **Remove unrelated**: Filter out titles that share the same type but are unrelated

**IMPORTANT**: Be INCLUSIVE rather than exclusive. If there's reasonable chance a title could help answer the query, keep it.

Examples of what to KEEP:
- Query: "airport named after Pat McCarran" → Keep: "McCarran International Airport", "Henderson Executive Airport" (reliever airport)
- Query: "2015 NHL Entry Draft" → Keep: "2015", "NHL Entry Draft", "National Hockey League" (fragments that together answer)
- Query: "ghost man slang" → Keep: "Gweilo", "Ghosts (2006 film)" (film about the slang term)

Examples of what to FILTER OUT:
- Query: "Argentine education" → Remove: "Education in Morocco", "Education in Greece" (same type, different country)
- Query: "Stephen Graham 2006 film" → Remove: "Stephen Wade" (same name, wrong person)

---

**INPUT:**

Query: {{QUERY}}

Entity: {{ENTITY_NAME}}
Type: {{ENTITY_TYPE}}
Subtype: {{ENTITY_SUBTYPE}}

Candidate Titles ({{COUNT}} total):
{{CANDIDATE_TITLES}}

---

**OUTPUT FORMAT (JSON only):**

{
  "relevant_titles": ["title1", "title2", ...],
  "filtered_out_titles": ["title3", "title4", ...],
  "reasoning": "Brief explanation of filtering decisions"
}

**Rules:**
- Include a title in "relevant_titles" if it could help answer the query
- Be inclusive - when in doubt, keep it
- Provide concise reasoning for major filtering decisions
"""


async def extract_entities_from_query(client: AsyncOpenAI, query: str) -> Dict:
    """
    Extract entities from a query using LLM.
    
    Args:
        client: AsyncOpenAI client
        query: The question/query string
        
    Returns:
        Dict with 'success', 'entities', and optional 'error'
    """
    try:
        formatted_prompt = ENTITY_EXTRACTION_PROMPT.replace("__QUESTION__", query)
        
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Remove code block markers if present
        if result_text.startswith('```json'):
            result_text = result_text[7:]
        if result_text.startswith('```'):
            result_text = result_text[3:]
        if result_text.endswith('```'):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        result = json.loads(result_text)
        
        return {
            'success': True,
            'entities': result.get('entities', [])
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'entities': []
        }


def stage1a_value_matching(
    db: MetadataDB,
    entity_name: str,
    entity_type: Optional[str] = None,
    entity_subtype: Optional[str] = None,
    use_fts: bool = True
) -> List[Dict]:
    """
    Stage 1-A: Value-based entity matching.
    Search for entity_name in all metadata values using FTS.
    
    Args:
        db: MetadataDB instance
        entity_name: Entity name to search for
        entity_type: Optional entity type (used for type fallback)
        entity_subtype: Optional entity subtype
        use_fts: Use FTS for search
        
    Returns:
        List of matched passages
    """
    if use_fts:
        matches = db.search_by_entity_fts(entity_name, entity_type, entity_subtype)
    else:
        matches = db.search_by_entity(entity_name, entity_type, entity_subtype, search_title_only=False)
    
    # Type fallback: if no matches with type, retry without type
    if len(matches) == 0 and entity_type:
        if use_fts:
            matches = db.search_by_entity_fts(entity_name, None, None)
        else:
            matches = db.search_by_entity(entity_name, None, None, search_title_only=False)
    
    return matches


async def stage1b_type_filtering(
    client: AsyncOpenAI,
    db: MetadataDB,
    query: str,
    entity_name: str,
    possible_types: List[Dict]  # Changed: now accepts list of {type, subtype} dicts
) -> Tuple[List[Dict], Dict]:
    """
    Stage 1-B: Type/Subtype filtering → LLM title filtering.
    Now supports multiple type/subtype combinations.
    
    1. Get all passages with matching type/subtype (try all possibilities)
    2. Extract titles only
    3. Use LLM to filter titles
    4. Return filtered passages
    
    Args:
        client: AsyncOpenAI client
        db: MetadataDB instance
        query: Original query
        entity_name: Entity name
        possible_types: List of {type, subtype} dicts to try
        
    Returns:
        Tuple of (filtered_passages, filter_info)
    """
    # Get candidates by trying all type/subtype combinations
    all_candidates = []
    tried_types = []
    
    if possible_types:
        for type_info in possible_types:
            entity_type = type_info.get('type')
            entity_subtype = type_info.get('subtype')
            tried_types.append(f"{entity_type}/{entity_subtype}")
            
            if entity_type:
                candidates = db.search_by_type(entity_type, entity_subtype)
                # Add to all_candidates (deduplicate by title)
                for cand in candidates:
                    if not any(c['title'] == cand['title'] for c in all_candidates):
                        all_candidates.append(cand)
    
    if not all_candidates:
        return [], {
            'stage': '1-B',
            'type_candidates': 0,
            'llm_filtered': 0,
            'tried_types': tried_types,
            'skipped': len(tried_types) == 0
        }
    
    # Extract titles only (no full metadata)
    candidate_titles = [c['title'] for c in all_candidates]
    
    # Format prompt for LLM (use first type as primary)
    primary_type = possible_types[0] if possible_types else {}
    prompt = TITLE_FILTERING_PROMPT.replace('{{QUERY}}', query)
    prompt = prompt.replace('{{ENTITY_NAME}}', entity_name)
    prompt = prompt.replace('{{ENTITY_TYPE}}', primary_type.get('type', 'Unknown'))
    prompt = prompt.replace('{{ENTITY_SUBTYPE}}', primary_type.get('subtype', 'Unknown'))
    prompt = prompt.replace('{{COUNT}}', str(len(candidate_titles)))
    prompt = prompt.replace('{{CANDIDATE_TITLES}}', json.dumps(candidate_titles, indent=2, ensure_ascii=False))
    
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2048
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        if result_text.startswith('```json'):
            result_text = result_text[7:]
        if result_text.startswith('```'):
            result_text = result_text[3:]
        if result_text.endswith('```'):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        result = json.loads(result_text)
        relevant_titles = set(result.get('relevant_titles', []))
        
        # Filter candidates by relevant titles
        filtered_passages = [c for c in all_candidates if c['title'] in relevant_titles]
        
        filter_info = {
            'stage': '1-B',
            'type_candidates': len(all_candidates),
            'llm_filtered': len(filtered_passages),
            'tried_types': tried_types,
            'reasoning': result.get('reasoning', '')
        }
        
        return filtered_passages, filter_info
        
    except Exception as e:
        # Fallback: return all type-matched candidates if LLM fails
        filter_info = {
            'stage': '1-B',
            'type_candidates': len(all_candidates),
            'llm_filtered': len(all_candidates),
            'tried_types': tried_types,
            'error': str(e),
            'fallback': True
        }
        return all_candidates, filter_info
        filter_info = {
            'stage': '1-B',
            'type_candidates': len(candidates),
            'llm_filtered': len(candidates),
            'error': str(e),
            'fallback': True
        }
        return candidates, filter_info


def stage2_merge_results(
    value_matches: List[Dict],
    type_matches: List[Dict]
) -> List[Dict]:
    """
    Stage 2: Merge results from Stage 1-A and Stage 1-B.
    Remove duplicates based on title.
    Add 'source' tag to track which stage contributed each passage.
    
    Args:
        value_matches: Results from value-based matching
        type_matches: Results from type-based LLM filtering
        
    Returns:
        Merged list of unique passages with 'source' tag
    """
    # Use dict to avoid duplicates (by title)
    merged = {}
    
    for passage in value_matches:
        passage['source'] = 'stage1a_value'
        merged[passage['title']] = passage
    
    for passage in type_matches:
        # If already exists from stage1a, mark as 'both'
        if passage['title'] in merged:
            merged[passage['title']]['source'] = 'both'
        else:
            passage['source'] = 'stage1b_type'
            merged[passage['title']] = passage
    
    return list(merged.values())


async def retrieve_for_entity_hybrid(
    client: AsyncOpenAI,
    db: MetadataDB,
    query: str,
    entity: Dict,
    use_fts: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Hybrid retrieval for a single entity.
    
    Pipeline:
      1-A: Value-based matching
      1-B: Type-based LLM filtering (tries multiple type/subtype combinations)
      2: Merge results
    
    Args:
        client: AsyncOpenAI client
        db: MetadataDB instance
        query: Original query
        entity: Entity dict with 'entity_name', 'possible_types' (list of {type, subtype})
        use_fts: Use FTS for value matching
        
    Returns:
        Tuple of (final_passages, retrieval_info)
    """
    entity_name = entity.get('entity_name')
    possible_types = entity.get('possible_types', [])
    
    # For backwards compatibility, support old format with single type/subtype
    if not possible_types and entity.get('type'):
        possible_types = [{
            'type': entity.get('type'),
            'subtype': entity.get('subtype')
        }]
    
    # Get first type for value matching (used for fallback)
    primary_type = possible_types[0] if possible_types else {}
    entity_type = primary_type.get('type')
    entity_subtype = primary_type.get('subtype')
    
    # Run Stage 1-A (sync) and 1-B (async) - Stage 1-A runs first due to SQLite thread limitation
    value_matches = stage1a_value_matching(db, entity_name, entity_type, entity_subtype, use_fts)
    type_matches, type_info = await stage1b_type_filtering(client, db, query, entity_name, possible_types)
    
    # Stage 2: Merge
    final_passages = stage2_merge_results(value_matches, type_matches)
    
    retrieval_info = {
        'entity_name': entity_name,
        'entity_type': entity_type,
        'entity_subtype': entity_subtype,
        'stage1a_value_matches': len(value_matches),
        'stage1b_type_info': type_info,
        'stage2_final': len(final_passages)
    }
    
    return final_passages, retrieval_info


async def retrieve_for_query(
    client: AsyncOpenAI,
    db: MetadataDB,
    query: str,
    use_fts: bool = True
) -> Dict:
    """
    Main hybrid retrieval function for a query with role-based entity processing.
    
    Args:
        client: AsyncOpenAI client for entity extraction and LLM filtering
        db: MetadataDB instance
        query: The original query
        use_fts: If True, use FTS for faster search
        
    Returns:
        Dict with extracted entities, retrieved passages and info
    """
    # Step 1: Extract entities from query using LLM (with roles)
    extraction_result = await extract_entities_from_query(client, query)
    
    if not extraction_result['success']:
        return {
            'query': query,
            'extraction_result': extraction_result,
            'extracted_entities': [],
            'retrieved_passages': [],
            'retrieval_info': {
                'total_entities': 0,
                'entity_results': [],
                'error': 'Entity extraction failed'
            }
        }
    
    extracted_entities = extraction_result['entities']
    
    # Step 2: Group entities by role
    target_entities = [e for e in extracted_entities if e.get('role') == 'target']
    attribute_entities = [e for e in extracted_entities if e.get('role') == 'attribute']
    context_entities = [e for e in extracted_entities if e.get('role') == 'context']
    
    # Step 3: Retrieve passages with role-based strategy
    all_passages = []
    entity_results = []
    
    # Strategy 1: Process all CRITICAL target entities (parallel for comparison)
    critical_targets = [e for e in target_entities if e.get('importance') == 'critical']
    
    if len(critical_targets) > 1:
        # Comparison question: retrieve all targets in parallel
        tasks = [retrieve_for_entity_hybrid(client, db, query, e, use_fts) for e in critical_targets]
        results = await asyncio.gather(*tasks)
        
        for passages, info in results:
            entity_results.append(info)
            for passage in passages:
                if not any(p['title'] == passage['title'] for p in all_passages):
                    all_passages.append(passage)
    
    elif len(critical_targets) == 1:
        # Single target question
        passages, info = await retrieve_for_entity_hybrid(client, db, query, critical_targets[0], use_fts)
        entity_results.append(info)
        all_passages.extend(passages)
    
    # Strategy 2: If insufficient results, try important target entities
    if len(all_passages) < 2:
        important_targets = [e for e in target_entities if e.get('importance') == 'important']
        for entity in important_targets:
            passages, info = await retrieve_for_entity_hybrid(client, db, query, entity, use_fts)
            entity_results.append(info)
            for passage in passages:
                if not any(p['title'] == passage['title'] for p in all_passages):
                    all_passages.append(passage)
    
    # Strategy 3: If still insufficient, try attribute entities
    if len(all_passages) < 2 and attribute_entities:
        for entity in attribute_entities[:2]:  # Limit to top 2 attributes
            passages, info = await retrieve_for_entity_hybrid(client, db, query, entity, use_fts)
            entity_results.append(info)
            for passage in passages:
                if not any(p['title'] == passage['title'] for p in all_passages):
                    all_passages.append(passage)
    
    # Strategy 4: Last resort - context entities
    if len(all_passages) < 1 and context_entities:
        for entity in context_entities[:1]:  # Only use first context
            passages, info = await retrieve_for_entity_hybrid(client, db, query, entity, use_fts)
            entity_results.append(info)
            for passage in passages:
                if not any(p['title'] == passage['title'] for p in all_passages):
                    all_passages.append(passage)
    
    return {
        'query': query,
        'extraction_result': extraction_result,
        'extracted_entities': extracted_entities,
        'retrieved_passages': all_passages,
        'retrieval_info': {
            'total_entities': len(extracted_entities),
            'target_entities': len(target_entities),
            'attribute_entities': len(attribute_entities),
            'context_entities': len(context_entities),
            'entity_results': entity_results,
            'total_unique_passages': len(all_passages)
        }
    }


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize LLM client and database
        print("Initializing LLM client and database...")
        client = initialize_llm_client()
        db = MetadataDB('metadata_v2.db')
        
        # Verify database connection
        stats = db.get_stats()
        print(f"Database loaded: {stats['total_entries']} passages")
        type_items = list(stats['type_distribution'].items())[:5]
        type_summary = ', '.join([f"{t}({c})" for t, c in type_items])
        print(f"Types: {type_summary}...")
        
        # Test queries
        test_queries = [
            "Stephen Graham starred in a film in 2006, directed by whom?",
            "Who proposed plan in which education in state institutions of Argentina is free?",
            "The Bee Cliff in northeast Tennessee overlooks a river that is how many miles long?",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"Test {i}: {query}")
            print('='*80)
            
            result = await retrieve_for_query(client, db, query, use_fts=True)
            
            # Display extraction
            print(f"\n[Entity Extraction]")
            if result['extraction_result']['success']:
                # Group by role
                targets = [e for e in result['extracted_entities'] if e.get('role') == 'target']
                attributes = [e for e in result['extracted_entities'] if e.get('role') == 'attribute']
                contexts = [e for e in result['extracted_entities'] if e.get('role') == 'context']
                
                if targets:
                    print(f"  Targets ({len(targets)}):")
                    for ent in targets:
                        print(f"    - {ent['entity_name']} ({ent.get('type', 'Unknown')}/{ent.get('subtype', 'Unknown')}) [{ent.get('importance', 'unknown')}]")
                
                if attributes:
                    print(f"  Attributes ({len(attributes)}):")
                    for ent in attributes:
                        print(f"    - {ent['entity_name']} ({ent.get('type', 'Unknown')}/{ent.get('subtype', 'Unknown')}) [{ent.get('importance', 'unknown')}]")
                
                if contexts:
                    print(f"  Context ({len(contexts)}):")
                    for ent in contexts:
                        print(f"    - {ent['entity_name']} ({ent.get('type', 'Unknown')}/{ent.get('subtype', 'Unknown')}) [{ent.get('importance', 'unknown')}]")
            else:
                print(f"  ✗ Failed: {result['extraction_result'].get('error')}")
            
            # Display retrieval info
            print(f"\n[Retrieval Results]")
            print(f"  Total entities: {result['retrieval_info']['total_entities']} (T:{result['retrieval_info']['target_entities']}, A:{result['retrieval_info']['attribute_entities']}, C:{result['retrieval_info']['context_entities']})")
            
            for ent_info in result['retrieval_info']['entity_results']:
                print(f"\n  Entity: {ent_info['entity_name']}")
                print(f"    Stage 1-A (Value): {ent_info['stage1a_value_matches']} passages")
                
                type_info = ent_info['stage1b_type_info']
                if type_info.get('skipped'):
                    print(f"    Stage 1-B (Type): Skipped (no type)")
                else:
                    print(f"    Stage 1-B (Type): {type_info['type_candidates']} candidates → {type_info['llm_filtered']} after LLM")
                    if 'reasoning' in type_info and type_info['reasoning']:
                        print(f"      Reasoning: {type_info['reasoning'][:100]}...")
                
                print(f"    Stage 2 (Merged): {ent_info['stage2_final']} final passages")
            
            # Display final passages
            print(f"\n[Final Passages] ({result['retrieval_info']['total_unique_passages']} unique)")
            for passage in result['retrieved_passages'][:5]:  # Show first 5
                print(f"  ✓ {passage['title']}")
            if len(result['retrieved_passages']) > 5:
                print(f"  ... and {len(result['retrieved_passages']) - 5} more")
        
        print(f"\n{'='*80}\n")
        db.close()
    
    asyncio.run(main())
