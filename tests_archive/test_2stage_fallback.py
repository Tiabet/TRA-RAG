"""
Test 2-stage fallback filtering with Leonard Logsdail
Full pipeline: Query Decomposition → Entity Extraction → Retrieval → Answer Generation
"""
import asyncio
import json
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from multihop_pipeline import process_single_question
from metadata_db import MetadataDB
from llm_logger import init_logger, finalize_log

# Load environment variables
load_dotenv()

async def test_fallback_filtering():
    """Test if 2-stage fallback (truncated -> full) works
    Full pipeline: Query Decomposition → Entity Extraction → Retrieval → Answer Generation
    """
    
    # Initialize logger
    init_logger()
    
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    db = MetadataDB('metadata_v2.db')
    
    print("\n" + "="*80)
    print("Testing 2-Stage Fallback Filtering - FULL PIPELINE")
    print("="*80)
    
    question_data = {
        "_id": "test_logsdail",
        "question": "Leonard Logsdail had a cameo role in the biographical film directed by whom?",
        "answer": "Martin Scorsese"
    }
    
    print(f"\nQuestion: {question_data['question']}")
    print(f"Gold Answer: {question_data['answer']}")
    
    print(f"\nPipeline:")
    print("  1. Query Decomposition (split into sub-queries)")
    print("  2. Entity Extraction (extract entities from each query)")
    print("  3. Hybrid Retrieval (2-stage fallback filtering)")
    print("  4. Answer Generation (with retrieved context)")
    
    # Run full pipeline
    print("\n" + "-"*80)
    print("Running Full Pipeline...")
    print("-"*80)
    
    result = await process_single_question(
        client=client,
        db=db,
        question_data=question_data,
        use_fts=True,
        apply_llm_filter_stage1a=True,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("PIPELINE RESULT:")
    print("="*80)
    
    if result['success']:
        print(f"✓ Pipeline completed successfully")
        print(f"\nDecomposition:")
        decomp = result.get('decomposition_result', {})
        if decomp.get('success'):
            print(f"  Question Type: {decomp.get('decomposition', {}).get('question_type', 'N/A')}")
            print(f"  Sub-questions: {len(decomp.get('decomposition', {}).get('subquestions', []))}")
        
        print(f"\nAnswering:")
        answering = result.get('answering_result', {})
        if answering.get('success'):
            subq_results = answering.get('subquestion_results', [])
            print(f"  Answered {len(subq_results)} sub-questions")
            
            # Check for 2-stage fallback usage
            fallback_used = False
            for sq_result in subq_results:
                for entity_name, retrieval_info in sq_result.get('retrieval_info', {}).items():
                    if retrieval_info.get('stage1a_value_info', {}).get('used_full_metadata_fallback'):
                        print(f"  ✓✓✓ FALLBACK used in Stage 1-A for '{entity_name}'")
                        fallback_used = True
                    if retrieval_info.get('stage1b_type_info', {}).get('used_full_metadata_fallback'):
                        print(f"  ✓✓✓ FALLBACK used in Stage 1-B for '{entity_name}'")
                        fallback_used = True
            
            if not fallback_used:
                print(f"  No full metadata fallback was triggered")
        
        print(f"\nFinal Answer:")
        final_answer = result.get('final_answer', 'No answer')
        print(f"  Predicted: {final_answer}")
        print(f"  Gold: {question_data['answer']}")
        
        is_correct = question_data['answer'].lower() in final_answer.lower()
        print(f"\n{'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        
        print(f"\nTiming:")
        print(f"  Total: {result.get('total_time', 0):.2f}s")
        print(f"  Decomposition: {result.get('decomposition_time', 0):.2f}s")
        print(f"  Answering: {result.get('answering_time', 0):.2f}s")
        print(f"  Synthesis: {result.get('synthesis_time', 0):.2f}s")
    else:
        print(f"✗ Pipeline failed")
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Finalize log
    log_file = finalize_log()
    print(f"\n📄 All LLM interactions logged to: {log_file}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_fallback_filtering())
