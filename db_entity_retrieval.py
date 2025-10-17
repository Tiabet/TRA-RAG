"""
Database-based Entity Retrieval System
======================================
Uses SQLite for efficient metadata search across all values.
"""
import asyncio
import os
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
        import json
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


def retrieve_passages_for_entities(
    db: MetadataDB,
    entities: List[Dict],
    use_fts: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Retrieve passages for multiple entities using database.
    
    Args:
        db: MetadataDB instance
        entities: List of entity dicts with 'entity_name', 'type', 'subtype'
        use_fts: If True, use FTS for faster search
        
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
        
        # Search using database
        if use_fts:
            matches = db.search_by_entity_fts(
                entity_name,
                entity_type,
                entity_subtype
            )
        else:
            matches = db.search_by_entity(
                entity_name,
                entity_type,
                entity_subtype,
                search_title_only=False  # Search in ALL values
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
            if not any(p['title'] == match['title'] for p in all_passages):
                all_passages.append(match)
    
    return all_passages, retrieval_info


async def retrieve_for_query(
    client: AsyncOpenAI,
    db: MetadataDB,
    query: str,
    use_fts: bool = True
) -> Dict:
    """
    Main retrieval function for a query with real-time entity extraction.
    
    Args:
        client: AsyncOpenAI client for entity extraction
        db: MetadataDB instance
        query: The original query
        use_fts: If True, use FTS for faster search
        
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
        db,
        extracted_entities,
        use_fts
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
    async def main():
        # Initialize LLM client
        print("Initializing LLM client...")
        client = initialize_llm_client()
        
        # Initialize database
        print("Opening database...")
        db = MetadataDB('metadata.db')
        
        try:
            # Test case 1: Baltic Cup (should find in title)
            print("\n" + "="*60)
            print("Test 1: Baltic Cup (in title)")
            print("="*60)
            
            query1 = "Which country refrained from participating in the 1991 Baltic Cup?"
            result1 = await retrieve_for_query(client, db, query1)
            
            print(f"Query: {result1['query']}")
            print(f"Extraction success: {result1['extraction_result']['success']}")
            print(f"Extracted entities: {[e['entity_name'] for e in result1['extracted_entities']]}")
            print(f"Retrieved passages: {len(result1['retrieved_passages'])}")
            for i, passage in enumerate(result1['retrieved_passages'][:5]):
                print(f"  {i+1}. {passage['title']}")
            
            # Test case 2: Entity in nested values (not in title)
            print("\n" + "="*60)
            print("Test 2: Estonia (in nested attributes/relations)")
            print("="*60)
            
            query2 = "Tell me about passages related to Estonia"
            result2 = await retrieve_for_query(client, db, query2)
            
            print(f"Query: {result2['query']}")
            print(f"Extracted entities: {[e['entity_name'] for e in result2['extracted_entities']]}")
            print(f"Retrieved passages: {len(result2['retrieved_passages'])}")
            for i, passage in enumerate(result2['retrieved_passages'][:5]):
                print(f"  {i+1}. {passage['title']}")
            
            # Test case 3: Compare FTS vs regular search
            print("\n" + "="*60)
            print("Test 3: Performance comparison")
            print("="*60)
            
            entity = {"entity_name": "Baltic Cup", "type": None, "subtype": None}
            
            import time
            
            # FTS search
            start = time.time()
            results_fts = db.search_by_entity_fts("Baltic Cup")
            time_fts = time.time() - start
            
            # Regular search
            start = time.time()
            results_regular = db.search_by_entity("Baltic Cup", search_title_only=False)
            time_regular = time.time() - start
            
            print(f"FTS search: {len(results_fts)} results in {time_fts*1000:.2f}ms")
            print(f"Regular search: {len(results_regular)} results in {time_regular*1000:.2f}ms")
            print(f"FTS speedup: {time_regular/time_fts:.1f}x faster")
            
        finally:
            db.close()
            print("\n✓ Database closed")
    
    # Run async main
    asyncio.run(main())
