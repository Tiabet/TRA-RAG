"""
Entity-based Metadata Retrieval for Multi-hop RAG
=================================================
Retrieves relevant passages using entity extraction and metadata matching.
"""
import json
import re
import os
import asyncio
from typing import List, Dict, Optional, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the entity extraction prompt
# Import the entity extraction prompt
from Prompt.entity_extraction_prompt import ENTITY_EXTRACTION_PROMPT


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
        
    except json.JSONDecodeError as e:
        return {
            'success': False,
            'error': f'JSON parse error: {str(e)}',
            'raw_response': result_text if 'result_text' in locals() else None,
            'entities': []
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'entities': []
        }


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison:
    - Lowercase
    - Remove special characters except spaces
    - Remove extra whitespace
    """
    # Lowercase
    text = text.lower()
    # Remove special characters except spaces and alphanumeric
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def title_contains_entity(title: str, entity_name: str) -> bool:
    """
    Check if title contains entity after normalization.
    
    Args:
        title: The metadata title
        entity_name: The extracted entity name
        
    Returns:
        True if normalized title contains normalized entity
    """
    norm_title = normalize_text(title)
    norm_entity = normalize_text(entity_name)
    
    return norm_entity in norm_title


def find_matching_metadata(
    metadata_list: List[Dict],
    entity_name: str,
    entity_type: Optional[str] = None,
    entity_subtype: Optional[str] = None
) -> List[Dict]:
    """
    Find metadata entries where title contains the entity.
    
    Args:
        metadata_list: List of metadata entries with 'title' and 'metadata'
        entity_name: Entity name to search for
        entity_type: Optional entity type for filtering
        entity_subtype: Optional entity subtype for filtering
        
    Returns:
        List of matching metadata entries
    """
    # Step 1: Find all metadata where title contains entity
    title_matches = []
    for entry in metadata_list:
        if title_contains_entity(entry['title'], entity_name):
            title_matches.append(entry)
    
    # If no matches, return empty list
    if not title_matches:
        return []
    
    # If only one match, return it
    if len(title_matches) == 1:
        return title_matches
    
    # If multiple matches, filter by type/subtype if provided
    if entity_type and entity_subtype:
        type_filtered = []
        for entry in title_matches:
            metadata = entry['metadata']
            
            # Check title-level type first (for passages without main_entities)
            if (metadata.get('type', '').lower() == entity_type.lower() and
                metadata.get('subtype', '').lower() == entity_subtype.lower()):
                type_filtered.append(entry)
                continue
            
            # Also check if any main_entity matches the type and subtype
            if 'main_entities' in metadata:
                for main_entity in metadata['main_entities']:
                    if (main_entity.get('type', '').lower() == entity_type.lower() and
                        main_entity.get('subtype', '').lower() == entity_subtype.lower()):
                        type_filtered.append(entry)
                        break  # Only add once per entry
        
        # If type filtering reduces to 1 or 0, return that
        if len(type_filtered) <= 1:
            return type_filtered
        
        # Otherwise return all title matches
        return title_matches
    
    # If type not provided, return all title matches
    return title_matches


def retrieve_passages_for_entities(
    metadata_list: List[Dict],
    entities: List[Dict]
) -> Tuple[List[Dict], Dict]:
    """
    Retrieve passages for multiple entities.
    
    Args:
        metadata_list: List of all metadata entries
        entities: List of entity dicts with 'entity_name', 'type', 'subtype'
        
    Returns:
        Tuple of (retrieved_passages, retrieval_info)
    """
    all_passages = []
    retrieval_info = {
        'total_entities': len(entities),
        'entity_results': []
    }
    
    for entity in entities:
        entity_name = entity.get('entity_name')
        entity_type = entity.get('type')
        entity_subtype = entity.get('subtype')
        
        matches = find_matching_metadata(
            metadata_list,
            entity_name,
            entity_type,
            entity_subtype
        )
        
        entity_info = {
            'entity_name': entity_name,
            'entity_type': entity_type,
            'entity_subtype': entity_subtype,
            'matches_found': len(matches),
            'status': 'success' if len(matches) > 0 else 'no_match'
        }
        
        retrieval_info['entity_results'].append(entity_info)
        
        # Add matches to all_passages (avoid duplicates)
        for match in matches:
            # Check if already in all_passages
            if not any(p['title'] == match['title'] for p in all_passages):
                all_passages.append(match)
    
    return all_passages, retrieval_info


async def retrieve_for_query(
    client: AsyncOpenAI,
    metadata_list: List[Dict],
    query: str
) -> Dict:
    """
    Main retrieval function for a query with real-time entity extraction.
    
    Args:
        client: AsyncOpenAI client for entity extraction
        metadata_list: List of all metadata entries
        query: The original query
        
    Returns:
        Dict with extracted entities, retrieved passages and info
    """
    # Step 1: Extract entities from query using LLM
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
    
    # Step 2: Retrieve passages based on extracted entities
    passages, retrieval_info = retrieve_passages_for_entities(
        metadata_list,
        extracted_entities
    )
    
    return {
        'query': query,
        'extraction_result': extraction_result,
        'extracted_entities': extracted_entities,
        'retrieved_passages': passages,
        'retrieval_info': retrieval_info
    }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize LLM client
        print("Initializing LLM client...")
        client = initialize_llm_client()
        
        # Load metadata
        print("Loading metadata...")
        with open('HotpotQA/hotpotqa_sample_200_pure_metadata.json', 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        print(f"Loaded {len(metadata_list)} metadata entries")
        
        # Test case 1: Single entity
        print("\n" + "="*60)
        print("Test 1: Single Entity (Real-time Extraction)")
        print("="*60)
        
        query1 = "The Bee Cliff in northeast Tennessee overlooks a river that is how many miles long?"
        result1 = await retrieve_for_query(client, metadata_list, query1)
        
        print(f"Query: {result1['query']}")
        print(f"Extraction success: {result1['extraction_result']['success']}")
        print(f"Extracted entities: {result1['extracted_entities']}")
        print(f"Retrieved passages: {len(result1['retrieved_passages'])}")
        for i, passage in enumerate(result1['retrieved_passages'][:3]):
            print(f"\n  {i+1}. {passage['title']}")
        
        # Test case 2: Multiple entities (comparison)
        print("\n" + "="*60)
        print("Test 2: Multiple Entities - Comparison (Real-time Extraction)")
        print("="*60)
        
        query2 = "Are both Stephen R. Donaldson and Michael Moorcock science fiction writers?"
        result2 = await retrieve_for_query(client, metadata_list, query2)
        
        print(f"Query: {result2['query']}")
        print(f"Extraction success: {result2['extraction_result']['success']}")
        print(f"Extracted entities: {[e['entity_name'] for e in result2['extracted_entities']]}")
        print(f"Retrieved passages: {len(result2['retrieved_passages'])}")
        for i, passage in enumerate(result2['retrieved_passages']):
            print(f"\n  {i+1}. {passage['title']}")
        
        # Test case 3: Custom query
        print("\n" + "="*60)
        print("Test 3: Custom Query (Real-time Extraction)")
        print("="*60)
        
        query3 = "Who is the founder of Tesla Motors?"
        result3 = await retrieve_for_query(client, metadata_list, query3)
        
        print(f"Query: {result3['query']}")
        print(f"Extraction success: {result3['extraction_result']['success']}")
        print(f"Extracted entities: {result3['extracted_entities']}")
        print(f"Retrieved passages: {len(result3['retrieved_passages'])}")
        for i, passage in enumerate(result3['retrieved_passages'][:3]):
            print(f"\n  {i+1}. {passage['title']}")
        
        print("\n" + "="*60)
        print("Retrieval Info:")
        print("="*60)
        for entity_result in result2['retrieval_info']['entity_results']:
            print(f"  {entity_result['entity_name']}: {entity_result['matches_found']} matches ({entity_result['status']})")
    
    # Run async main
    asyncio.run(main())
