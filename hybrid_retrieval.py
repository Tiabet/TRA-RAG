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
from Prompt.title_filtering_prompt import TITLE_FILTERING_PROMPT
from llm_logger import log_llm_call, log_llm_error

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
        
        # Log LLM interaction
        log_llm_call(
            call_type="Entity Extraction (Main Query)",
            input_text=formatted_prompt,
            output_text=result_text,
            context={"query": query}
        )
        
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
        log_llm_error(
            call_type="Entity Extraction (Main Query)",
            error=str(e),
            context={"query": query}
        )
        return {
            'success': False,
            'error': str(e),
            'entities': []
        }


async def stage1a_value_matching(
    client: AsyncOpenAI,
    db: MetadataDB,
    query: str,
    entity_name: str,
    entity_type: Optional[str] = None,
    entity_subtype: Optional[str] = None,
    use_fts: bool = True,
    apply_llm_filter: bool = True,
    main_query: Optional[str] = None
) -> Tuple[List[Dict], Dict]:
    """
    Stage 1-A: Value-based entity matching with optional LLM filtering.
    Search for entity_name in all metadata values using FTS, then optionally filter with LLM.
    
    Args:
        client: AsyncOpenAI client
        db: MetadataDB instance
        query: Current sub-query (for LLM filtering)
        entity_name: Entity name to search for
        entity_type: Optional entity type (used for type fallback)
        entity_subtype: Optional entity subtype
        use_fts: Use FTS for search
        apply_llm_filter: If True, apply LLM title filtering (NEW!)
        main_query: Optional main query for dual-query filtering
        
    Returns:
        Tuple of (matched_passages, filter_info)
    """
    # Step 1: Get initial matches from DB
    # IMPORTANT: Stage 1-A is VALUE-BASED search, so we DON'T apply type filters
    # Type filtering is done in Stage 1-B (type-based LLM filtering)
    if use_fts:
        matches = db.search_by_entity_fts(entity_name, None, None)  # No type filter
    else:
        matches = db.search_by_entity(entity_name, None, None, search_title_only=False)
    
    initial_count = len(matches)
    
    # Step 2: Apply LLM filtering if requested (with 2-stage fallback)
    if apply_llm_filter and matches:
        def prepare_candidates(matches, use_full_metadata=False):
            """Prepare candidates with metadata snippets
            
            Args:
                matches: List of matched passages
                use_full_metadata: If True, include full metadata without truncation
                                  If False, use truncated version for efficiency
            """
            candidates = []
            for m in matches:
                # Use matched_fields if available (from FTS search)
                matched_fields = m.get('matched_fields', [])
                
                if matched_fields and not use_full_metadata:
                    # Fast mode: Show matched key-value pairs with truncation
                    matched_info = []
                    for field in matched_fields[:5]:  # Max 5 matched fields
                        key = field['key']
                        value = field['value']
                        # Truncate long values
                        if len(value) > 150:
                            value = value[:150] + "..."
                        matched_info.append(f"{key}: {value}")
                    
                    snippet = '\n'.join(matched_info) if matched_info else 'No matched fields'
                else:
                    # Full mode or fallback: Extract metadata
                    metadata_str = m.get('metadata', '')
                    if isinstance(metadata_str, str):
                        try:
                            metadata = json.loads(metadata_str)
                        except:
                            metadata = {}
                    else:
                        metadata = metadata_str or {}
                    
                    # Build snippet from metadata fields
                    snippet_parts = []
                    excluded_keys = {'title', 'type', 'subtype'}
                    
                    if use_full_metadata:
                        # FULL MODE: No truncation, all fields
                        for key, value in metadata.items():
                            if key not in excluded_keys and value:
                                value_str = str(value)
                                snippet_parts.append(f"{key}: {value_str}")
                    else:
                        # FAST MODE: Truncated, priority fields only
                        priority_keys = ['relations', 'events', 'attributes', 'description']
                        for key in priority_keys:
                            if key in metadata and metadata[key]:
                                value_str = str(metadata[key])
                                # relations: keep full, others: truncate
                                if key == 'relations':
                                    snippet_parts.append(f"{key}: {value_str}")
                                else:
                                    if len(value_str) > 200:
                                        value_str = value_str[:200] + "..."
                                    snippet_parts.append(f"{key}: {value_str}")
                    
                    snippet = '\n'.join(snippet_parts) if snippet_parts else 'No metadata'
                
                candidates.append({
                    'title': m['title'],
                    'type': m.get('type', 'Unknown'),
                    'subtype': m.get('subtype', 'Unknown'),
                    'matched_content': snippet
                })
            return candidates
        
        # STAGE 1: Try with truncated metadata (fast)
        candidates_with_snippets = prepare_candidates(matches, use_full_metadata=False)
        
        # Format prompt for LLM
        prompt = TITLE_FILTERING_PROMPT.replace('{{MAIN_QUERY}}', main_query or query)
        prompt = prompt.replace('{{SUB_QUERY}}', query)
        prompt = prompt.replace('{{ENTITY_NAME}}', entity_name)
        prompt = prompt.replace('{{ENTITY_TYPE}}', entity_type or 'Unknown')
        prompt = prompt.replace('{{ENTITY_SUBTYPE}}', entity_subtype or 'Unknown')
        prompt = prompt.replace('{{COUNT}}', str(len(candidates_with_snippets)))
        prompt = prompt.replace('{{CANDIDATES}}', json.dumps(candidates_with_snippets, indent=2, ensure_ascii=False))
        
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
            
            # Log LLM interaction
            log_llm_call(
                call_type="LLM Title Filtering (Stage 1-A, Fast Mode)",
                input_text=prompt,
                output_text=result_text,
                context={
                    "query": query,
                    "entity_name": entity_name,
                    "num_candidates": len(candidates_with_snippets),
                    "mode": "truncated_metadata"
                }
            )
            
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
            reasoning = result.get('reasoning', '')
            
            # STAGE 2: Fallback if insufficient information detected
            needs_retry = (
                not relevant_titles or  # No results
                'insufficient' in reasoning.lower() or  # Explicit insufficient
                'not enough' in reasoning.lower() or
                'need more' in reasoning.lower() or
                'unclear' in reasoning.lower()
            )
            
            if needs_retry and len(matches) > 0:
                # Retry with full metadata
                candidates_full = prepare_candidates(matches, use_full_metadata=True)
                
                prompt_full = TITLE_FILTERING_PROMPT.replace('{{MAIN_QUERY}}', main_query or query)
                prompt_full = prompt_full.replace('{{SUB_QUERY}}', query)
                prompt_full = prompt_full.replace('{{ENTITY_NAME}}', entity_name)
                prompt_full = prompt_full.replace('{{ENTITY_TYPE}}', entity_type or 'Unknown')
                prompt_full = prompt_full.replace('{{ENTITY_SUBTYPE}}', entity_subtype or 'Unknown')
                prompt_full = prompt_full.replace('{{COUNT}}', str(len(candidates_full)))
                prompt_full = prompt_full.replace('{{CANDIDATES}}', json.dumps(candidates_full, indent=2, ensure_ascii=False))
                
                response_full = await client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": prompt_full}
                    ],
                    temperature=0.1,
                    max_tokens=2048
                )
                
                result_text_full = response_full.choices[0].message.content.strip()
                
                # Log fallback LLM interaction
                log_llm_call(
                    call_type="LLM Title Filtering (Stage 1-A, Full Metadata Fallback)",
                    input_text=prompt_full,
                    output_text=result_text_full,
                    context={
                        "query": query,
                        "entity_name": entity_name,
                        "num_candidates": len(candidates_full),
                        "mode": "full_metadata",
                        "reason": "insufficient_information_in_fast_mode"
                    }
                )
                
                # Parse fallback response
                if result_text_full.startswith('```json'):
                    result_text_full = result_text_full[7:]
                if result_text_full.startswith('```'):
                    result_text_full = result_text_full[3:]
                if result_text_full.endswith('```'):
                    result_text_full = result_text_full[:-3]
                result_text_full = result_text_full.strip()
                
                result = json.loads(result_text_full)
                relevant_titles = set(result.get('relevant_titles', []))
                
                # Mark that fallback was used
                used_fallback = True
            else:
                used_fallback = False
            
            # Filter matches by relevant titles
            filtered_matches = [m for m in matches if m['title'] in relevant_titles]
            
            filter_info = {
                'stage': '1-A',
                'initial_matches': initial_count,
                'llm_filtered': len(filtered_matches),
                'reasoning': result.get('reasoning', ''),
                'used_full_metadata_fallback': used_fallback
            }
            
            return filtered_matches, filter_info
            
        except Exception as e:
            log_llm_error(
                call_type="LLM Title Filtering (Stage 1-A)",
                error=str(e),
                context={
                    "query": query,
                    "entity_name": entity_name
                }
            )
            # Fallback: return all matches if LLM fails
            filter_info = {
                'stage': '1-A',
                'initial_matches': initial_count,
                'llm_filtered': initial_count,
                'error': str(e),
                'fallback': True
            }
            return matches, filter_info
    
    # No LLM filtering applied
    filter_info = {
        'stage': '1-A',
        'initial_matches': initial_count,
        'llm_filtered': initial_count,
        'llm_filter_applied': False
    }
    
    return matches, filter_info


async def stage1b_type_filtering(
    client: AsyncOpenAI,
    db: MetadataDB,
    query: str,
    entity_name: str,
    possible_types: List[Dict],  # Changed: now accepts list of {type, subtype} dicts
    main_query: Optional[str] = None
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
        query: Current sub-query
        entity_name: Entity name
        possible_types: List of {type, subtype} dicts to try
        main_query: Optional main query for dual-query filtering
        
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
    
    # Helper function to prepare candidates (same as Stage 1-A)
    def prepare_candidates_1b(candidates, use_full_metadata=False):
        """Prepare candidates with metadata snippets
        
        Args:
            candidates: List of candidate passages
            use_full_metadata: If True, include full metadata without truncation
        """
        result = []
        for c in candidates:
            # Extract metadata
            metadata_str = c.get('metadata', '')
            if isinstance(metadata_str, str):
                try:
                    metadata = json.loads(metadata_str)
                except:
                    metadata = {}
            else:
                metadata = metadata_str or {}
            
            # Build matched content from metadata fields
            content_parts = []
            excluded_keys = {'title', 'type', 'subtype'}
            
            if use_full_metadata:
                # FULL MODE: No truncation, all fields
                for key, value in metadata.items():
                    if key not in excluded_keys and value:
                        value_str = str(value)
                        content_parts.append(f"{key}: {value_str}")
            else:
                # FAST MODE: Truncated, priority fields only
                priority_keys = ['relations', 'events', 'attributes', 'description']
                for key in priority_keys:
                    if key in metadata and metadata[key]:
                        value_str = str(metadata[key])
                        # relations: keep full, others: truncate
                        if key == 'relations':
                            content_parts.append(f"{key}: {value_str}")
                        else:
                            if len(value_str) > 200:
                                value_str = value_str[:200] + "..."
                            content_parts.append(f"{key}: {value_str}")
            
            matched_content = '\n'.join(content_parts) if content_parts else 'No metadata available'
            
            result.append({
                'title': c['title'],
                'type': c.get('type', 'Unknown'),
                'subtype': c.get('subtype', 'Unknown'),
                'matched_content': matched_content
            })
        return result
    
    # STAGE 1: Try with truncated metadata (fast)
    candidates_with_snippets = prepare_candidates_1b(all_candidates, use_full_metadata=False)
    
    # Format prompt for LLM (use first type as primary)
    primary_type = possible_types[0] if possible_types else {}
    prompt = TITLE_FILTERING_PROMPT.replace('{{MAIN_QUERY}}', main_query or query)
    prompt = prompt.replace('{{SUB_QUERY}}', query)
    prompt = prompt.replace('{{ENTITY_NAME}}', entity_name)
    prompt = prompt.replace('{{ENTITY_TYPE}}', primary_type.get('type', 'Unknown'))
    prompt = prompt.replace('{{ENTITY_SUBTYPE}}', primary_type.get('subtype', 'Unknown'))
    prompt = prompt.replace('{{COUNT}}', str(len(candidates_with_snippets)))
    prompt = prompt.replace('{{CANDIDATES}}', json.dumps(candidates_with_snippets, indent=2, ensure_ascii=False))
    
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
        
        # Log LLM interaction
        log_llm_call(
            call_type="LLM Title Filtering (Stage 1-B, Fast Mode)",
            input_text=prompt,
            output_text=result_text,
            context={
                "query": query,
                "entity_name": entity_name,
                "num_candidates": len(candidates_with_snippets),
                "types": [f"{t.get('type', '')}/{t.get('subtype', '')}" for t in possible_types],
                "mode": "truncated_metadata"
            }
        )
        
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
        reasoning = result.get('reasoning', '')
        
        # STAGE 2: Fallback if insufficient information detected
        needs_retry = (
            not relevant_titles or  # No results
            'insufficient' in reasoning.lower() or  # Explicit insufficient
            'not enough' in reasoning.lower() or
            'need more' in reasoning.lower() or
            'unclear' in reasoning.lower()
        )
        
        if needs_retry and len(all_candidates) > 0:
            # Retry with full metadata
            candidates_full = prepare_candidates_1b(all_candidates, use_full_metadata=True)
            
            prompt_full = TITLE_FILTERING_PROMPT.replace('{{MAIN_QUERY}}', main_query or query)
            prompt_full = prompt_full.replace('{{SUB_QUERY}}', query)
            prompt_full = prompt_full.replace('{{ENTITY_NAME}}', entity_name)
            prompt_full = prompt_full.replace('{{ENTITY_TYPE}}', primary_type.get('type', 'Unknown'))
            prompt_full = prompt_full.replace('{{ENTITY_SUBTYPE}}', primary_type.get('subtype', 'Unknown'))
            prompt_full = prompt_full.replace('{{COUNT}}', str(len(candidates_full)))
            prompt_full = prompt_full.replace('{{CANDIDATES}}', json.dumps(candidates_full, indent=2, ensure_ascii=False))
            
            response_full = await client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt_full}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            
            result_text_full = response_full.choices[0].message.content.strip()
            
            # Log fallback LLM interaction
            log_llm_call(
                call_type="LLM Title Filtering (Stage 1-B, Full Metadata Fallback)",
                input_text=prompt_full,
                output_text=result_text_full,
                context={
                    "query": query,
                    "entity_name": entity_name,
                    "num_candidates": len(candidates_full),
                    "types": [f"{t.get('type', '')}/{t.get('subtype', '')}" for t in possible_types],
                    "mode": "full_metadata",
                    "reason": "insufficient_information_in_fast_mode"
                }
            )
            
            # Parse fallback response
            if result_text_full.startswith('```json'):
                result_text_full = result_text_full[7:]
            if result_text_full.startswith('```'):
                result_text_full = result_text_full[3:]
            if result_text_full.endswith('```'):
                result_text_full = result_text_full[:-3]
            result_text_full = result_text_full.strip()
            
            result = json.loads(result_text_full)
            relevant_titles = set(result.get('relevant_titles', []))
            
            # Mark that fallback was used
            used_fallback = True
        else:
            used_fallback = False
        
        # Filter candidates by relevant titles
        filtered_passages = [c for c in all_candidates if c['title'] in relevant_titles]
        
        filter_info = {
            'stage': '1-B',
            'type_candidates': len(all_candidates),
            'llm_filtered': len(filtered_passages),
            'tried_types': tried_types,
            'reasoning': result.get('reasoning', ''),
            'used_full_metadata_fallback': used_fallback
        }
        
        return filtered_passages, filter_info
        
    except Exception as e:
        log_llm_error(
            call_type="LLM Title Filtering (Stage 1-B)",
            error=str(e),
            context={
                "query": query,
                "entity_name": entity_name,
                "types": [f"{t.get('type', '')}/{t.get('subtype', '')}" for t in possible_types]
            }
        )
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
    use_fts: bool = True,
    apply_llm_filter_stage1a: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Hybrid retrieval for a single entity.
    
    Pipeline:
      1-A: Value-based matching (+ optional LLM filtering)
      1-B: Type-based LLM filtering (tries multiple type/subtype combinations)
      2: Merge results
    
    Args:
        client: AsyncOpenAI client
        db: MetadataDB instance
        query: Original query
        entity: Entity dict with 'entity_name', 'possible_types' (list of {type, subtype})
        use_fts: Use FTS for value matching
        apply_llm_filter_stage1a: If True, apply LLM title filtering to Stage 1-A (NEW!)
        
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
    
    # Run Stage 1-A (now async with LLM filtering) and 1-B in parallel
    value_matches, value_info = await stage1a_value_matching(
        client, db, query, entity_name, entity_type, entity_subtype, use_fts, apply_llm_filter_stage1a
    )
    type_matches, type_info = await stage1b_type_filtering(client, db, query, entity_name, possible_types)
    
    # Stage 2: Merge
    final_passages = stage2_merge_results(value_matches, type_matches)
    
    retrieval_info = {
        'entity_name': entity_name,
        'entity_type': entity_type,
        'entity_subtype': entity_subtype,
        'stage1a_value_info': value_info,
        'stage1b_type_info': type_info,
        'stage2_final': len(final_passages)
    }
    
    return final_passages, retrieval_info


async def retrieve_for_query(
    client: AsyncOpenAI,
    db: MetadataDB,
    query: str,
    use_fts: bool = True,
    apply_llm_filter_stage1a: bool = True
) -> Dict:
    """
    Main hybrid retrieval function for a query with role-based entity processing.
    
    Args:
        client: AsyncOpenAI client for entity extraction and LLM filtering
        db: MetadataDB instance
        query: The original query
        use_fts: If True, use FTS for faster search
        apply_llm_filter_stage1a: If True, apply LLM title filtering to Stage 1-A (NEW!)
        
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
        tasks = [retrieve_for_entity_hybrid(client, db, query, e, use_fts, apply_llm_filter_stage1a) for e in critical_targets]
        results = await asyncio.gather(*tasks)
        
        for passages, info in results:
            entity_results.append(info)
            for passage in passages:
                if not any(p['title'] == passage['title'] for p in all_passages):
                    all_passages.append(passage)
    
    elif len(critical_targets) == 1:
        # Single target question
        passages, info = await retrieve_for_entity_hybrid(client, db, query, critical_targets[0], use_fts, apply_llm_filter_stage1a)
        entity_results.append(info)
        all_passages.extend(passages)
    
    # Strategy 2: If insufficient results, try important target entities
    if len(all_passages) < 2:
        important_targets = [e for e in target_entities if e.get('importance') == 'important']
        for entity in important_targets:
            passages, info = await retrieve_for_entity_hybrid(client, db, query, entity, use_fts, apply_llm_filter_stage1a)
            entity_results.append(info)
            for passage in passages:
                if not any(p['title'] == passage['title'] for p in all_passages):
                    all_passages.append(passage)
    
    # Strategy 3: If still insufficient, try attribute entities
    if len(all_passages) < 2 and attribute_entities:
        for entity in attribute_entities[:2]:  # Limit to top 2 attributes
            passages, info = await retrieve_for_entity_hybrid(client, db, query, entity, use_fts, apply_llm_filter_stage1a)
            entity_results.append(info)
            for passage in passages:
                if not any(p['title'] == passage['title'] for p in all_passages):
                    all_passages.append(passage)
    
    # Strategy 4: Last resort - context entities
    if len(all_passages) < 1 and context_entities:
        for entity in context_entities[:1]:  # Only use first context
            passages, info = await retrieve_for_entity_hybrid(client, db, query, entity, use_fts, apply_llm_filter_stage1a)
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
                
                # Stage 1-A info (now with LLM filtering)
                value_info = ent_info['stage1a_value_info']
                if value_info.get('llm_filter_applied', True):
                    print(f"    Stage 1-A (Value): {value_info['initial_matches']} initial → {value_info['llm_filtered']} after LLM")
                    if 'reasoning' in value_info and value_info['reasoning']:
                        print(f"      Reasoning: {value_info['reasoning'][:100]}...")
                else:
                    print(f"    Stage 1-A (Value): {value_info['initial_matches']} passages (no LLM filter)")
                
                # Stage 1-B info
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
