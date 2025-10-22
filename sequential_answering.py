"""
Sequential Answering Module
=============================
Answers sub-questions sequentially using hybrid retrieval and LLM generation.

Pipeline for each sub-question:
1. Substitute [SQ{N}_Answer] placeholders with actual answers
2. Build context from previous sub-question answers
3. Extract entities using sub-question-specific prompt
4. Retrieve passages using hybrid retrieval (Stage 1-A + 1-B)
5. Generate answer from passages using LLM
"""

import json
import asyncio
from typing import Dict, List, Optional
from openai import AsyncOpenAI

from metadata_db import MetadataDB
from query_decomposition import (
    SubQuestion,
    QueryDecomposition,
    substitute_answers,
    build_context_from_previous
)
from Prompt.subquestion_entity_extraction_prompt import (
    SUBQUESTION_ENTITY_EXTRACTION_PROMPT
)
from Prompt.subquestion_answering_prompt import (
    SUBQUESTION_ANSWERING_PROMPT,
    FINAL_ANSWER_SYNTHESIS_PROMPT
)


async def extract_entities_from_subquestion(
    client: AsyncOpenAI,
    subquestion: str,
    previous_context: str = ""
) -> Dict:
    """
    Extract entities from a sub-question using specialized prompt.
    
    Args:
        client: AsyncOpenAI client
        subquestion: The sub-question text
        previous_context: Context from previous sub-questions
        
    Returns:
        Dict with 'success', 'entities', and optional 'error'
    """
    try:
        # Format prompt
        formatted_prompt = SUBQUESTION_ENTITY_EXTRACTION_PROMPT.replace(
            "__SUBQUESTION__", 
            subquestion
        )
        
        # Add previous context
        if previous_context:
            formatted_prompt = formatted_prompt.replace(
                "{{previous_context}}", 
                previous_context
            )
        else:
            formatted_prompt = formatted_prompt.replace(
                "{{previous_context}}", 
                "(None - this is the first sub-question)"
            )
        
        # Call LLM
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


async def retrieve_for_subquestion(
    client: AsyncOpenAI,
    db: MetadataDB,
    subquestion: str,
    entities: List[Dict],
    use_fts: bool = True,
    apply_llm_filter_stage1a: bool = True
) -> Dict:
    """
    Retrieve passages for a sub-question using hybrid retrieval.
    
    Reuses the existing hybrid_retrieval logic:
    - Stage 1-A: Value/FTS matching + LLM filtering
    - Stage 1-B: Type filtering + LLM filtering
    - Stage 2: Merge and deduplicate
    
    Args:
        client: AsyncOpenAI client
        db: MetadataDB instance
        subquestion: The sub-question text
        entities: Extracted entities with types
        use_fts: Use FTS for Stage 1-A
        apply_llm_filter_stage1a: Apply LLM filtering to Stage 1-A
        
    Returns:
        Dict with 'passages', 'retrieval_info'
    """
    from hybrid_retrieval import (
        stage1a_value_matching,
        stage1b_type_filtering,
        stage2_merge_results
    )
    
    all_passages = []
    retrieval_info_list = []
    
    # Process each entity
    for entity in entities:
        entity_name = entity.get('entity_name', '')
        possible_types = entity.get('possible_types', [])
        
        if not entity_name:
            continue
        
        # Get primary type
        primary_type = possible_types[0] if possible_types else {}
        entity_type = primary_type.get('type')
        entity_subtype = primary_type.get('subtype')
        
        # Run Stage 1-A and 1-B in parallel
        value_matches, value_info = await stage1a_value_matching(
            client, db, subquestion, entity_name, 
            entity_type, entity_subtype, 
            use_fts, apply_llm_filter_stage1a
        )
        
        type_matches, type_info = await stage1b_type_filtering(
            client, db, subquestion, entity_name, possible_types
        )
        
        # Stage 2: Merge
        entity_passages = stage2_merge_results(value_matches, type_matches)
        
        all_passages.extend(entity_passages)
        
        retrieval_info_list.append({
            'entity_name': entity_name,
            'entity_type': entity_type,
            'entity_subtype': entity_subtype,
            'stage1a_value_info': value_info,
            'stage1b_type_info': type_info,
            'stage2_final': len(entity_passages)
        })
    
    # Deduplicate passages across entities (by title)
    seen_titles = set()
    unique_passages = []
    for passage in all_passages:
        title = passage.get('title', '')
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_passages.append(passage)
    
    return {
        'passages': unique_passages,
        'retrieval_info': retrieval_info_list
    }


async def generate_answer_from_passages(
    client: AsyncOpenAI,
    subquestion: str,
    passages: List[Dict],
    previous_context: str = ""
) -> str:
    """
    Generate answer from retrieved passages using LLM.
    
    Args:
        client: AsyncOpenAI client
        subquestion: The sub-question text
        passages: Retrieved passages
        previous_context: Context from previous sub-questions
        
    Returns:
        Generated answer string
    """
    try:
        # Format passages
        passage_texts = []
        for i, passage in enumerate(passages[:10], 1):  # Top 10 passages
            title = passage.get('title', 'Unknown')
            metadata = passage.get('metadata', {})
            
            # Extract key info from metadata
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            # Build passage text
            passage_parts = [f"[{i}] {title}"]
            
            for key in ['description', 'main_entity', 'attributes', 'events']:
                if key in metadata and metadata[key]:
                    value = str(metadata[key])[:200]  # First 200 chars
                    passage_parts.append(f"  {key}: {value}")
            
            passage_texts.append('\n'.join(passage_parts))
        
        passages_text = '\n\n'.join(passage_texts) if passage_texts else "No passages retrieved."
        
        # Format prompt
        prompt = SUBQUESTION_ANSWERING_PROMPT.replace(
            "{{subquestion}}", 
            subquestion
        )
        prompt = prompt.replace(
            "{{passages}}", 
            passages_text
        )
        prompt = prompt.replace(
            "{{previous_context}}", 
            previous_context if previous_context else "(None)"
        )
        
        # Call LLM
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise question answering system. Give short, direct answers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=100  # Short answers
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Clean up answer
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        
        return answer
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"


async def answer_subquestion(
    client: AsyncOpenAI,
    db: MetadataDB,
    sq: SubQuestion,
    decomposition: QueryDecomposition,
    use_fts: bool = True,
    apply_llm_filter_stage1a: bool = True
) -> Dict:
    """
    Answer a single sub-question using the full pipeline.
    
    Pipeline:
    1. Substitute [SQ{N}_Answer] placeholders
    2. Build context from previous answers
    3. Extract entities (sub-question specific)
    4. Retrieve passages (hybrid retrieval)
    5. Generate answer (LLM)
    
    Args:
        client: AsyncOpenAI client
        db: MetadataDB instance
        sq: SubQuestion object
        decomposition: QueryDecomposition with previous answers
        use_fts: Use FTS for retrieval
        apply_llm_filter_stage1a: Apply LLM filtering to Stage 1-A
        
    Returns:
        Dict with 'success', 'answer', 'passages', 'retrieval_info', optional 'error'
    """
    try:
        # Step 1: Substitute placeholders
        actual_question = substitute_answers(sq.question, decomposition.subquestions)
        
        # Step 2: Build context from previous answers
        previous_context = build_context_from_previous(sq, decomposition)
        
        # Step 3: Extract entities
        extraction_result = await extract_entities_from_subquestion(
            client, actual_question, previous_context
        )
        
        if not extraction_result['success']:
            return {
                'success': False,
                'error': f"Entity extraction failed: {extraction_result.get('error', 'Unknown error')}"
            }
        
        entities = extraction_result['entities']
        
        # Step 4: Retrieve passages
        retrieval_result = await retrieve_for_subquestion(
            client, db, actual_question, entities,
            use_fts, apply_llm_filter_stage1a
        )
        
        passages = retrieval_result['passages']
        retrieval_info = retrieval_result['retrieval_info']
        
        # Step 5: Generate answer
        answer = await generate_answer_from_passages(
            client, actual_question, passages, previous_context
        )
        
        # Update SubQuestion object
        sq.answer = answer
        sq.retrieved_passages = passages
        sq.retrieval_info = {
            'actual_question': actual_question,
            'extracted_entities': entities,
            'retrieval_info': retrieval_info,
            'num_passages': len(passages)
        }
        
        return {
            'success': True,
            'answer': answer,
            'actual_question': actual_question,
            'entities': entities,
            'passages': passages,
            'retrieval_info': retrieval_info
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


async def synthesize_final_answer(
    client: AsyncOpenAI,
    decomposition: QueryDecomposition
) -> str:
    """
    Synthesize final answer from all sub-question answers.
    
    Args:
        client: AsyncOpenAI client
        decomposition: QueryDecomposition with all answers filled
        
    Returns:
        Final answer string
    """
    try:
        # Build sub-question chain text
        chain_parts = []
        for sq in decomposition.subquestions:
            chain_parts.append(f"{sq.id}: {sq.question}")
            chain_parts.append(f"Answer: {sq.answer if sq.answer else '(Not answered)'}")
            chain_parts.append("")  # Empty line
        
        subquestion_chain = '\n'.join(chain_parts)
        
        # Format prompt
        prompt = FINAL_ANSWER_SYNTHESIS_PROMPT.replace(
            "{{main_question}}", 
            decomposition.main_query
        )
        prompt = prompt.replace(
            "{{subquestion_chain}}", 
            subquestion_chain
        )
        
        # Call LLM
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise question answering system. Give direct, concise final answers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        final_answer = response.choices[0].message.content.strip()
        
        # Clean up answer
        if final_answer.startswith("Final Answer:"):
            final_answer = final_answer[13:].strip()
        
        return final_answer
        
    except Exception as e:
        return f"Error synthesizing final answer: {str(e)}"


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from query_decomposition import decompose_query
    
    load_dotenv()
    
    async def test_sequential_answering():
        """Test sequential answering with a sample question"""
        
        # Initialize clients
        client = AsyncOpenAI(
            api_key=os.getenv('ALICE_OPENAI_KEY'),
            base_url=os.getenv('ALICE_CHAT_URL')
        )
        
        db_path = 'metadata_v2.db'
        if not os.path.exists(db_path):
            print(f"❌ Database not found: {db_path}")
            return
        
        db = MetadataDB(db_path)
        
        # Test question
        test_query = "The Bee Cliff in northeast Tennessee overlooks a river that is how many miles long?"
        
        print(f"\n{'='*80}")
        print(f"Testing Sequential Answering")
        print(f"{'='*80}")
        print(f"\nMain Query: {test_query}\n")
        
        # Step 1: Decompose
        print("Step 1: Query Decomposition...")
        decomp_result = await decompose_query(client, test_query)
        
        if not decomp_result['success']:
            print(f"❌ Decomposition failed: {decomp_result['error']}")
            return
        
        decomposition = decomp_result['decomposition']
        print(f"✅ Decomposed into {len(decomposition.subquestions)} sub-questions")
        
        for sq in decomposition.subquestions:
            print(f"  {sq.id}: {sq.question}")
        
        # Step 2: Answer each sub-question sequentially
        print(f"\n{'='*80}")
        print("Step 2: Sequential Answering...")
        print(f"{'='*80}\n")
        
        for sq in decomposition.subquestions:
            print(f"\n{'-'*80}")
            print(f"Answering {sq.id}: {sq.question}")
            print(f"{'-'*80}")
            
            result = await answer_subquestion(
                client, db, sq, decomposition,
                use_fts=True,
                apply_llm_filter_stage1a=True
            )
            
            if result['success']:
                print(f"✅ Answer: {result['answer']}")
                print(f"   Entities extracted: {len(result['entities'])}")
                print(f"   Passages retrieved: {len(result['passages'])}")
            else:
                print(f"❌ Error: {result['error']}")
        
        # Step 3: Synthesize final answer
        print(f"\n{'='*80}")
        print("Step 3: Final Answer Synthesis...")
        print(f"{'='*80}\n")
        
        final_answer = await synthesize_final_answer(client, decomposition)
        print(f"Final Answer: {final_answer}")
        
        # Close DB
        db.close()
    
    # Run test
    asyncio.run(test_sequential_answering())
