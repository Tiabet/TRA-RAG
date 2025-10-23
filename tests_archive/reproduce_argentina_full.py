"""
Argentina 케이스 전체 파이프라인을 재현합니다.
"""
import asyncio
import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from metadata_db import MetadataDB
from sequential_answering import retrieve_for_subquestion

load_dotenv()

async def reproduce_argentina_case():
    # Initialize
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    db = MetadataDB('metadata_v2.db')
    
    # Argentina case data
    main_query = "Who proposed plan in which education in state institutions of Argentina is free at the initial, primary, secondary and tertiary levels and in the undergraduate university level?"
    sub_query = "What is the plan for free education in state institutions of Argentina at the initial, primary, secondary, and tertiary levels?"
    
    # Extracted entities from checkpoint (SQ1)
    entities = [
        {
            'entity_name': 'free education',
            'possible_types': [
                {'type': 'Concept', 'subtype': 'EducationalSystem'},
                {'type': 'Concept', 'subtype': 'Policy'},
                {'type': 'Concept', 'subtype': 'SocialSystem'}
            ]
        },
        {
            'entity_name': 'state institutions',
            'possible_types': [
                {'type': 'Organization', 'subtype': 'EducationalInstitution'},
                {'type': 'Organization', 'subtype': 'GovernmentAgency'},
                {'type': 'Concept', 'subtype': 'AdministrativeUnit'}
            ]
        },
        {
            'entity_name': 'Argentina',
            'possible_types': [
                {'type': 'Location', 'subtype': 'Country'},
                {'type': 'Location', 'subtype': 'GeopoliticalEntity'}
            ]
        },
        {
            'entity_name': 'initial level',
            'possible_types': [
                {'type': 'Concept', 'subtype': 'EducationalLevel'},
                {'type': 'Concept', 'subtype': 'AcademicStage'},
                {'type': 'Concept', 'subtype': 'GradeLevel'}
            ]
        },
        {
            'entity_name': 'primary level',
            'possible_types': [
                {'type': 'Concept', 'subtype': 'EducationalLevel'},
                {'type': 'Concept', 'subtype': 'AcademicStage'},
                {'type': 'Concept', 'subtype': 'GradeLevel'}
            ]
        },
        {
            'entity_name': 'secondary level',
            'possible_types': [
                {'type': 'Concept', 'subtype': 'EducationalLevel'},
                {'type': 'Concept', 'subtype': 'AcademicStage'},
                {'type': 'Concept', 'subtype': 'GradeLevel'}
            ]
        },
        {
            'entity_name': 'tertiary level',
            'possible_types': [
                {'type': 'Concept', 'subtype': 'EducationalLevel'},
                {'type': 'Concept', 'subtype': 'AcademicStage'},
                {'type': 'Concept', 'subtype': 'GradeLevel'}
            ]
        }
    ]
    
    print("="*80)
    print("Argentina 케이스 전체 파이프라인 재현")
    print("="*80)
    print(f"\nMain Query: {main_query}")
    print(f"\nSub Query (SQ1): {sub_query}")
    print(f"\nExtracted Entities: {len(entities)}개")
    for e in entities:
        print(f"  - {e['entity_name']}")
    
    # Run retrieval
    print("\n" + "="*80)
    print("실제 Retrieval 실행 중...")
    print("="*80)
    
    result = await retrieve_for_subquestion(
        client, db, sub_query, entities,
        use_fts=True,
        apply_llm_filter_stage1a=True,
        main_query=main_query
    )
    
    passages = result['passages']
    retrieval_info = result['retrieval_info']
    
    print(f"\n✅ 최종 Passages: {len(passages)}개")
    for i, p in enumerate(passages, 1):
        print(f"  {i}. {p['title']}")
    
    print(f"\n📊 Entity별 Retrieval 상세:")
    for info in retrieval_info:
        entity_name = info['entity_name']
        stage1a = info['stage1a_value_info']
        stage1b = info['stage1b_type_info']
        stage2 = info['stage2_final']
        
        print(f"\n🎯 Entity: {entity_name}")
        print(f"  Stage 1-A (Value): {stage1a['initial_matches']}개 → {stage1a['llm_filtered']}개 (LLM filtered)")
        print(f"  Stage 1-B (Type): {stage1b.get('initial_matches', 0)}개 → {stage1b.get('llm_filtered', 0)}개 (LLM filtered)")
        print(f"  Stage 2 (Merged): {stage2}개")
    
    # Check if Taquini Plan is in final results
    if any('Taquini' in p['title'] for p in passages):
        print("\n✅ Taquini Plan이 최종 결과에 포함되었습니다!")
    else:
        print("\n❌ Taquini Plan이 최종 결과에 없습니다!")
        print("\n분석:")
        print("  • 'Argentina' entity 검색에서 나왔는지 확인이 필요합니다")
        print("  • LLM Filtering에서 제거되었는지 확인이 필요합니다")
    
    db.close()

if __name__ == "__main__":
    asyncio.run(reproduce_argentina_case())
