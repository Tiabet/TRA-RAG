"""
LLM-Filtered Entity Retrieval System
=====================================
2-stage filtering:
1. Type/Subtype matching (fast)
2. LLM semantic filtering (accurate)
"""
import asyncio
import json
import os
from typing import List, Dict, Optional, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv

from metadata_db import MetadataDB
from Prompt.entity_extraction_prompt import ENTITY_EXTRACTION_PROMPT
from Prompt.llm_filtering_prompt import LLM_FILTERING_PROMPT

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
        # Format the prompt with the question
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
        
        # Parse JSON
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


def create_candidate_snippet(metadata: Dict) -> str:
    """
    Create a concise snippet from metadata for LLM filtering.
    
    Args:
        metadata: The full metadata dict
        
    Returns:
        A concise string representation of key attributes
    """
    snippets = []
    
    # Add description if available
    if 'attributes' in metadata:
        attrs = metadata['attributes']
        
        # Common descriptive fields (expanded list)
        for key in ['description', 'full_name', 'summary', 'meaning', 'definition', 'title_reference']:
            if key in attrs:
                value = attrs[key]
                # Handle nested dicts (like title_reference)
                if isinstance(value, dict):
                    snippets.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
                else:
                    snippets.append(f"{key}: {value}")
        
        # Add a few other key attributes (increased from 3 to 5)
        other_keys = [k for k in attrs.keys() if k not in ['description', 'full_name', 'summary', 'meaning', 'definition', 'title_reference']]
        for key in other_keys[:5]:
            value = attrs[key]
            # Simplify complex nested structures
            if isinstance(value, dict):
                # Show first level of nested dict
                nested_str = ", ".join([f"{k}: {v}" for k, v in list(value.items())[:3]])
                snippets.append(f"{key}: {{{nested_str}}}")
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    snippets.append(f"{key}: [{len(value)} items]")
                else:
                    snippets.append(f"{key}: {value[:3]}")
            else:
                snippets.append(f"{key}: {value}")
    
    return "; ".join(snippets) if snippets else "No additional details"


async def llm_filter_candidates(
    client: AsyncOpenAI,
    query: str,
    entity_name: str,
    entity_type: str,
    entity_subtype: str,
    candidates: List[Dict]
) -> Dict:
    """
    Use LLM to filter candidate passages based on semantic relevance.
    
    Args:
        client: AsyncOpenAI client
        query: Original query
        entity_name: Extracted entity name
        entity_type: Entity type
        entity_subtype: Entity subtype
        candidates: List of candidate passages from DB
        
    Returns:
        Dict with 'relevant_passages' and 'filtered_out'
    """
    if not candidates:
        return {'relevant_passages': [], 'filtered_out': []}
    
    # Prepare candidates for prompt
    candidate_list = []
    for cand in candidates:
        snippet = create_candidate_snippet(cand['metadata'])
        candidate_list.append({
            'title': cand['title'],
            'type': cand['metadata'].get('type', 'Unknown'),
            'subtype': cand['metadata'].get('subtype', 'Unknown'),
            'snippet': snippet
        })
    
    # Format prompt
    prompt = LLM_FILTERING_PROMPT.replace('{{QUERY}}', query)
    prompt = prompt.replace('{{ENTITY_NAME}}', entity_name)
    prompt = prompt.replace('{{ENTITY_TYPE}}', entity_type or 'Unknown')
    prompt = prompt.replace('{{ENTITY_SUBTYPE}}', entity_subtype or 'Unknown')
    prompt = prompt.replace('{{CANDIDATES}}', json.dumps(candidate_list, indent=2, ensure_ascii=False))
    
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON
        result = json.loads(result_text)
        
        return {
            'success': True,
            'relevant_passages': result.get('relevant_passages', []),
            'filtered_out': result.get('filtered_out', [])
        }
        
    except Exception as e:
        # Fallback: return all candidates if LLM filtering fails
        return {
            'success': False,
            'error': str(e),
            'relevant_passages': [{'title': c['title'], 'confidence': 'unknown', 'reasoning': 'LLM filter failed'} for c in candidates],
            'filtered_out': []
        }


def retrieve_passages_for_entity_stage1(
    db: MetadataDB,
    entity_name: str,
    entity_type: Optional[str] = None,
    entity_subtype: Optional[str] = None,
    use_fts: bool = True
) -> List[Dict]:
    """
    Stage 1: Retrieve candidates using Type/Subtype filtering.
    
    Args:
        db: MetadataDB instance
        entity_name: Entity name to search for
        entity_type: Optional entity type
        entity_subtype: Optional entity subtype
        use_fts: Use FTS for search
        
    Returns:
        List of candidate passages
    """
    if use_fts:
        candidates = db.search_by_entity_fts(entity_name, entity_type, entity_subtype)
    else:
        candidates = db.search_by_entity(entity_name, entity_type, entity_subtype, search_title_only=False)
    
    return candidates


async def retrieve_passages_for_entity_stage2(
    client: AsyncOpenAI,
    db: MetadataDB,
    query: str,
    entity: Dict,
    use_fts: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    2-stage retrieval for a single entity.
    
    Args:
        client: AsyncOpenAI client
        db: MetadataDB instance
        query: Original query
        entity: Entity dict with 'entity_name', 'type', 'subtype'
        use_fts: Use FTS for stage 1
        
    Returns:
        Tuple of (final_passages, retrieval_info)
    """
    entity_name = entity.get('entity_name')
    entity_type = entity.get('type')
    entity_subtype = entity.get('subtype')
    
    # Stage 1: Type/Subtype filtering
    stage1_candidates = retrieve_passages_for_entity_stage1(
        db, entity_name, entity_type, entity_subtype, use_fts
    )
    
    retrieval_info = {
        'entity_name': entity_name,
        'entity_type': entity_type,
        'entity_subtype': entity_subtype,
        'stage1_candidates': len(stage1_candidates),
        'stage2_filtered': 0,
        'final_passages': 0
    }
    
    if not stage1_candidates:
        return [], retrieval_info
    
    # Stage 2: LLM filtering
    filter_result = await llm_filter_candidates(
        client, query, entity_name, entity_type, entity_subtype, stage1_candidates
    )
    
    if not filter_result.get('success', True):
        retrieval_info['llm_filter_error'] = filter_result.get('error')
    
    # Map relevant titles back to full passages
    relevant_titles = {p['title'] for p in filter_result.get('relevant_passages', [])}
    final_passages = [c for c in stage1_candidates if c['title'] in relevant_titles]
    
    retrieval_info['stage2_filtered'] = len(filter_result.get('filtered_out', []))
    retrieval_info['final_passages'] = len(final_passages)
    retrieval_info['llm_reasoning'] = filter_result.get('relevant_passages', [])
    
    return final_passages, retrieval_info


async def retrieve_for_query(
    client: AsyncOpenAI,
    db: MetadataDB,
    query: str,
    use_fts: bool = True
) -> Dict:
    """
    Main retrieval function with 2-stage filtering.
    
    Args:
        client: AsyncOpenAI client for entity extraction and filtering
        db: MetadataDB instance
        query: The original query
        use_fts: Use FTS for stage 1 search
        
    Returns:
        Dict with extracted entities, retrieved passages and detailed info
    """
    # Extract entities from query
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
    
    # Retrieve passages for each entity with 2-stage filtering
    all_passages = []
    entity_results = []
    
    for entity in extracted_entities:
        passages, info = await retrieve_passages_for_entity_stage2(
            client, db, query, entity, use_fts
        )
        
        entity_results.append(info)
        
        # Add to all_passages (deduplicate)
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
            'entity_results': entity_results
        }
    }


# Example usage
if __name__ == "__main__":
    async def main():
        print("Initializing LLM-Filtered Retrieval System...")
        client = initialize_llm_client()
        db = MetadataDB('metadata_v2.db')
        
        try:
            # Test case 1: Spelling variation (Roissy Airport)
            print("\n" + "="*80)
            print("Test 1: Spelling Variation - Roissy Airport")
            print("="*80)
            
            query1 = "The Roissy Airport connects to Paris and cities in what countries?"
            result1 = await retrieve_for_query(client, db, query1)
            
            print(f"Query: {result1['query']}")
            print(f"Extracted entities: {[e['entity_name'] for e in result1['extracted_entities']]}")
            print(f"\nRetrieval Process:")
            for entity_result in result1['retrieval_info']['entity_results']:
                print(f"\n  Entity: {entity_result['entity_name']}")
                print(f"  Stage 1 (Type filter): {entity_result['stage1_candidates']} candidates")
                print(f"  Stage 2 (LLM filter): {entity_result['final_passages']} final passages")
                if 'llm_reasoning' in entity_result:
                    for reasoning in entity_result['llm_reasoning']:
                        print(f"    ✓ {reasoning['title']} ({reasoning['confidence']}): {reasoning['reasoning'][:60]}...")
            
            print(f"\nFinal passages: {len(result1['retrieved_passages'])}")
            for p in result1['retrieved_passages']:
                print(f"  - {p['title']}")
            
            # Test case 2: Ghost man (Cantonese slang)
            print("\n" + "="*80)
            print("Test 2: Context Matching - Ghost Man / Gweilo")
            print("="*80)
            
            query2 = "What Cantonese slang term can mean both 'ghost man' and to refer to Westerners?"
            result2 = await retrieve_for_query(client, db, query2)
            
            print(f"Query: {result2['query']}")
            print(f"Extracted entities: {[e['entity_name'] for e in result2['extracted_entities']]}")
            print(f"\nRetrieval Process:")
            for entity_result in result2['retrieval_info']['entity_results']:
                print(f"\n  Entity: {entity_result['entity_name']}")
                print(f"  Stage 1 (Type filter): {entity_result['stage1_candidates']} candidates")
                print(f"  Stage 2 (LLM filter): {entity_result['final_passages']} final passages")
            
            print(f"\nFinal passages: {len(result2['retrieved_passages'])}")
            for p in result2['retrieved_passages']:
                print(f"  - {p['title']}")
            
        finally:
            db.close()
    
    asyncio.run(main())
