"""
Argentina Case - Complete Retrieval Trace
==========================================
Traces the ENTIRE retrieval process for the Argentina question from the checkpoint data.
Shows every step: entity extraction, FTS search, LLM filtering, and why Taquini Plan was missed.
"""

import json
import asyncio
import os
import sys
from dotenv import load_dotenv
from openai import AsyncOpenAI

from metadata_db import MetadataDB

load_dotenv()

# Force UTF-8 for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_checkpoint():
    """Load the checkpoint data"""
    with open('multihop_pipeline_200_checkpoint.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def find_argentina_case(data):
    """Find the Argentina question in checkpoint data"""
    for result in data['results']:
        if 'Argentina' in result['question'] and 'education' in result['question']:
            return result
    return None


async def trace_retrieval_process():
    """Trace the complete retrieval process for Argentina case"""
    
    print("="*100)
    print("ARGENTINA CASE - COMPLETE RETRIEVAL TRACE")
    print("="*100)
    
    # Load checkpoint data
    checkpoint_data = load_checkpoint()
    argentina_case = find_argentina_case(checkpoint_data)
    
    if not argentina_case:
        print("ERROR: Argentina case not found in checkpoint!")
        return
    
    # Display question info
    print(f"\n[QUESTION INFO]")
    print(f"Question ID: {argentina_case['question_id']}")
    print(f"Question: {argentina_case['question']}")
    print(f"Gold Answer: {argentina_case['gold_answer']}")
    print(f"Predicted Answer: {argentina_case['predicted_answer']}")
    print(f"Gold Supporting Facts: {argentina_case['gold_supporting_facts']}")
    
    # Display decomposition
    print(f"\n{'='*100}")
    print(f"[STEP 1] QUERY DECOMPOSITION")
    print(f"{'='*100}")
    
    decomp = argentina_case['decomposition']
    print(f"\nDetected Type: {decomp['detected_type']}")
    print(f"Number of Sub-Questions: {decomp['num_subquestions']}")
    
    for sq in decomp['subquestions']:
        print(f"\n  {sq['id']}: {sq['question']}")
        print(f"  Answer: {sq['answer']}")
        print(f"  Depends on: {sq['depends_on']}")
    
    # Display extracted entities
    print(f"\n{'='*100}")
    print(f"[STEP 2] ENTITY EXTRACTION")
    print(f"{'='*100}")
    
    entities = argentina_case['extracted_entities']
    print(f"\nTotal Entities Extracted: {entities['count']}")
    print(f"Unique Names: {entities['unique_names']}")
    
    # Group by subquestion
    sq1_entities = [e for e in entities['all'] if e.get('subquestion_id') == 'SQ1']
    sq2_entities = [e for e in entities['all'] if e.get('subquestion_id') == 'SQ2']
    
    print(f"\n[SQ1 Entities] ({len(sq1_entities)} entities)")
    for e in sq1_entities:
        types_str = ', '.join([f"{t.get('type', 'N/A')}/{t.get('subtype', 'N/A')}" for t in e.get('types', [])[:2]])
        print(f"  - '{e['name']}' [{e['role']}]: {types_str}")
    
    print(f"\n[SQ2 Entities] ({len(sq2_entities)} entities)")
    for e in sq2_entities:
        types_str = ', '.join([f"{t.get('type', 'N/A')}/{t.get('subtype', 'N/A')}" for t in e.get('types', [])[:2]])
        print(f"  - '{e['name']}' [{e['role']}]: {types_str}")
    
    # Display retrieved passages
    print(f"\n{'='*100}")
    print(f"[STEP 3] RETRIEVAL RESULTS")
    print(f"{'='*100}")
    
    passages = argentina_case['retrieved_passages']
    print(f"\nTotal Passages Retrieved: {passages['count']}")
    print(f"All Titles: {passages['titles']}")
    
    # Check if Taquini Plan is in results
    has_taquini = any('Taquini' in title for title in passages['titles'])
    print(f"\n*** Taquini Plan in results? {'YES' if has_taquini else 'NO'} ***")
    
    # Show per-subquestion retrieval
    for sq_passage in passages['by_subquestion']:
        sq_id = sq_passage['subquestion_id']
        titles = sq_passage['titles']
        print(f"\n[{sq_id}] Retrieved {len(titles)} passages:")
        for title in titles:
            marker = " <<< TAQUINI!" if 'Taquini' in title else ""
            print(f"  - {title}{marker}")
    
    # Now trace why Taquini Plan was not retrieved
    print(f"\n{'='*100}")
    print(f"[STEP 4] DETAILED TRACE - Why was Taquini Plan missed?")
    print(f"{'='*100}")
    
    # Initialize DB
    db = MetadataDB('metadata_v2.db')
    
    # Check if Taquini Plan exists
    print(f"\n[4.1] Checking if 'Taquini Plan' exists in database...")
    taquini_results = db.search_by_entity("Taquini Plan", None, None, search_title_only=False)
    print(f"      Found {len(taquini_results)} results")
    
    if taquini_results:
        for r in taquini_results:
            metadata_str = r.get('metadata', '')
            try:
                metadata_obj = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                desc = metadata_obj.get('description', 'N/A')
                print(f"      Title: {r['title']}")
                print(f"      Type: {r.get('type', 'N/A')} / Subtype: {r.get('subtype', 'N/A')}")
                print(f"      Description: {desc}")
            except:
                print(f"      Title: {r['title']} (metadata parse error)")
    
    # Test FTS search for each SQ1 entity
    print(f"\n[4.2] Testing FTS search for each SQ1 entity...")
    
    for e in sq1_entities:
        entity_name = e['name']
        print(f"\n      Entity: '{entity_name}'")
        
        fts_results = db.search_by_entity_fts(entity_name, None, None)
        print(f"      FTS Results: {len(fts_results)} passages")
        
        if fts_results:
            print(f"      Top 5 titles:")
            for i, r in enumerate(fts_results[:5], 1):
                marker = " <<< TAQUINI!" if 'Taquini' in r['title'] else ""
                print(f"        {i}. {r['title']}{marker}")
            
            # Check if Taquini is in results
            if any('Taquini' in r['title'] for r in fts_results):
                print(f"      *** Taquini Plan FOUND in FTS results for '{entity_name}' ***")
        else:
            print(f"      No FTS results")
    
    # Test FTS search for "Education in Argentina" specifically
    print(f"\n[4.3] Testing FTS search for 'Education in Argentina'...")
    edu_arg_results = db.search_by_entity_fts("Education in Argentina", None, None)
    print(f"      FTS Results: {len(edu_arg_results)} passages")
    
    if edu_arg_results:
        print(f"      All titles:")
        for r in edu_arg_results:
            marker = " <<< TAQUINI!" if 'Taquini' in r['title'] else ""
            print(f"        - {r['title']}{marker}")
    
    # Test what happens with "argentina" alone
    print(f"\n[4.4] Testing FTS search for 'argentina' (single word)...")
    arg_results = db.search_by_entity_fts("argentina", None, None)
    print(f"      FTS Results: {len(arg_results)} passages")
    
    if arg_results:
        print(f"      Titles containing 'Taquini':")
        taquini_found = False
        for r in arg_results:
            if 'Taquini' in r['title']:
                taquini_found = True
                print(f"        - {r['title']}")
        
        if not taquini_found:
            print(f"        (None)")
        
        print(f"      Top 5 titles:")
        for i, r in enumerate(arg_results[:5], 1):
            print(f"        {i}. {r['title']}")
    
    # Check type-based search
    print(f"\n[4.5] Testing type-based search for Concept/EducationalSystem...")
    type_results = db.search_by_type("Concept", "EducationalSystem")
    print(f"      Type Search Results: {len(type_results)} passages")
    
    if type_results:
        taquini_in_type = any('Taquini' in r['title'] for r in type_results)
        print(f"      Taquini Plan in type results? {'YES' if taquini_in_type else 'NO'}")
        
        if taquini_in_type:
            print(f"      Taquini Plan details:")
            for r in type_results:
                if 'Taquini' in r['title']:
                    print(f"        - {r['title']}")
                    print(f"          Type: {r.get('type')} / Subtype: {r.get('subtype')}")
    
    # Summary
    print(f"\n{'='*100}")
    print(f"[ANALYSIS SUMMARY]")
    print(f"{'='*100}")
    
    print(f"\n1. Gold Answer: '{argentina_case['gold_answer']}'")
    print(f"   Predicted Answer: '{argentina_case['predicted_answer']}'")
    
    print(f"\n2. Gold Supporting Facts:")
    for fact in argentina_case['gold_supporting_facts']:
        print(f"   - {fact[0]} (sentence {fact[1]})")
    
    print(f"\n3. Retrieved Passages:")
    print(f"   Total: {len(passages['titles'])} passages")
    print(f"   Has 'Taquini Plan'? {'YES' if has_taquini else 'NO'}")
    print(f"   Has 'Education in Argentina'? {'YES' if 'Education in Argentina' in passages['titles'] else 'NO'}")
    
    print(f"\n4. Possible Failure Reasons:")
    if not has_taquini:
        print(f"   [ ] Taquini Plan not in database (CHECKED: It exists)")
        print(f"   [?] Entity extraction didn't extract the right terms")
        print(f"   [?] FTS search didn't match 'Taquini Plan' with extracted entities")
        print(f"   [?] LLM filtering removed 'Taquini Plan'")
        print(f"   [?] Type matching failed (wrong type/subtype in DB)")
    
    db.close()
    
    print(f"\n{'='*100}")
    print(f"TRACE COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    asyncio.run(trace_retrieval_process())
