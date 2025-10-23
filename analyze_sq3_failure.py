"""
Analyze SQ3 Failure for "Seven years before" case
==================================================
Check what passages were retrieved for SQ3 and why it failed.
"""
import asyncio
import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI

from metadata_db import MetadataDB
from query_decomposition import decompose_query
from sequential_answering import answer_subquestions_sequential

load_dotenv()


async def analyze_sq3_failure():
    """Analyze why SQ3 failed to answer."""
    
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    db = MetadataDB('metadata_v2.db')
    
    main_query = "Seven years before the opening of the Brewer Fieldhouse in Columbia, Missouri, what was a campus of the University of Missouri known as?"
    
    print("=" * 80)
    print("Analyzing SQ3 Failure: Seven Years Before Case")
    print("=" * 80)
    print(f"Main Query: {main_query}\n")
    
    # Decompose
    decomposition_result = await decompose_query(client, main_query)
    decomposition = decomposition_result['decomposition']
    
    # Answer sub-questions
    answer_result = await answer_subquestions_sequential(
        client, db, decomposition,
        use_fts=True,
        apply_llm_filter_stage1a=True,
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("Detailed Analysis")
    print("=" * 80)
    
    for sq in decomposition.subquestions:
        print(f"\n{'=' * 80}")
        print(f"{sq.id}: {sq.question}")
        print(f"{'=' * 80}")
        print(f"Answer: {sq.answer}")
        
        if hasattr(sq, 'retrieval_info'):
            info = sq.retrieval_info
            print(f"\nActual Question (after substitution): {info.get('actual_question', 'N/A')}")
            
            entities = info.get('extracted_entities', [])
            print(f"\nExtracted Entities ({len(entities)}):")
            for ent in entities:
                print(f"  - {ent.get('name', 'Unknown')}")
                types = ent.get('types', [])
                if types:
                    for t in types[:2]:
                        print(f"    Type: {t.get('type', 'Unknown')}/{t.get('subtype', 'Unknown')}")
        
        if hasattr(sq, 'retrieved_passages'):
            passages = sq.retrieved_passages
            print(f"\nRetrieved Passages ({len(passages)}):")
            for i, p in enumerate(passages, 1):
                title = p.get('title', 'Unknown')
                metadata = p.get('metadata', {})
                
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                print(f"\n  [{i}] {title}")
                
                # Show key metadata fields
                for key in ['description', 'main_entity', 'attributes']:
                    if key in metadata and metadata[key]:
                        value = str(metadata[key])
                        # Show first 300 chars
                        if len(value) > 300:
                            value = value[:300] + "..."
                        print(f"      {key}: {value}")
        
        # Special attention to SQ3
        if sq.id == "SQ3":
            print("\n" + "=" * 80)
            print("SQ3 FAILURE ANALYSIS")
            print("=" * 80)
            
            # Check if "University Farm" appears in any passage
            if hasattr(sq, 'retrieved_passages'):
                found_in_passages = []
                for p in sq.retrieved_passages:
                    title = p.get('title', '')
                    metadata = p.get('metadata', {})
                    
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    
                    # Check if "University Farm" or "1922" appears
                    full_text = json.dumps(metadata).lower()
                    if "university farm" in full_text or "farm campus" in full_text:
                        found_in_passages.append(title)
                
                if found_in_passages:
                    print(f"✅ 'University Farm' FOUND in passages: {found_in_passages}")
                    print("   → LLM failed to extract it!")
                else:
                    print("❌ 'University Farm' NOT found in retrieved passages")
                    print("   → Retrieval problem!")
            
            # Check previous context
            from query_decomposition import build_context_from_previous
            prev_context = build_context_from_previous(sq, decomposition)
            
            print(f"\nPrevious Context Length: {len(prev_context)} chars")
            if "University Farm" in prev_context or "1922" in prev_context:
                print("✅ Relevant info found in previous context!")
            else:
                print("⚠️  Previous context may not have needed info")
    
    # Check DB directly for "University Farm" and "1922"
    print("\n" + "=" * 80)
    print("Direct DB Check")
    print("=" * 80)
    
    # Search for "University Farm"
    results = db.search_by_entity_fts("University Farm", None, None)
    print(f"\nDB Search 'University Farm': {len(results)} results")
    for r in results[:3]:
        print(f"  - {r.get('title', 'Unknown')}")
    
    # Search for "University of Missouri"
    results = db.search_by_entity_fts("University of Missouri", None, None)
    print(f"\nDB Search 'University of Missouri': {len(results)} results")
    for r in results[:5]:
        title = r.get('title', 'Unknown')
        metadata = r.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        desc = metadata.get('description', '')
        if '1922' in str(desc) or 'farm' in str(desc).lower():
            print(f"  ⭐ {title} (contains '1922' or 'farm')")
        else:
            print(f"  - {title}")
    
    # Search for "1922"
    results = db.search_by_entity_fts("1922", None, None)
    print(f"\nDB Search '1922': {len(results)} results")
    for r in results[:5]:
        title = r.get('title', 'Unknown')
        if 'missouri' in title.lower() or 'university' in title.lower():
            print(f"  ⭐ {title}")
        else:
            print(f"  - {title}")
    
    db.close()


if __name__ == "__main__":
    asyncio.run(analyze_sq3_failure())
