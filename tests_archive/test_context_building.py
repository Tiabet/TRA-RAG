"""
Argentina 케이스에서 Previous Context가 제대로 전달되는지 테스트
"""
import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from metadata_db import MetadataDB
from query_decomposition import QueryDecomposition, SubQuestion, build_context_from_previous

load_dotenv()

async def test_context_building():
    # Mock decomposition with SQ1 already answered
    decomposition = QueryDecomposition(
        main_query="Who proposed plan in which education in state institutions of Argentina is free...",
        question_type="bridge",
        reasoning="Two-step bridge question",
        subquestions=[]
    )
    
    # SQ1 (already answered with passages)
    sq1 = SubQuestion(
        id="SQ1",
        question="What is the plan for free education in state institutions of Argentina at the initial, primary, secondary, and tertiary levels?",
        depends_on=[],
        reasoning="Identify the specific education plan"
    )
    sq1.answer = "Free at initial, primary, secondary, and tertiary levels."
    
    # Mock retrieved passages for SQ1
    sq1.retrieved_passages = [
        {
            'title': 'Taquini Plan',
            'type': 'Concept',
            'subtype': 'EducationalReform',
            'metadata': {
                'description': 'A project for the restructuring of higher education in Argentina',
                'main_entity': 'Dr. Alberto Taquini',
                'attributes': {
                    'proposed_by': 'Dr. Alberto Taquini',
                    'year': '1968'
                }
            }
        },
        {
            'title': 'Education in Argentina',
            'type': 'Concept',
            'subtype': 'SocialSystem',
            'metadata': {
                'description': 'Education in state institutions of Argentina is free at the initial, primary, secondary and tertiary levels',
                'main_entity': 'Education system'
            }
        },
        {
            'title': 'Free education',
            'type': 'Concept',
            'subtype': 'EducationalSystem',
            'metadata': {
                'description': 'A system where education is funded by government and free for students'
            }
        }
    ]
    
    decomposition.subquestions.append(sq1)
    
    # SQ2 (depends on SQ1)
    sq2 = SubQuestion(
        id="SQ2",
        question="Who proposed [SQ1_Answer]?",
        depends_on=["SQ1"],
        reasoning="Find the person who proposed the identified plan"
    )
    
    decomposition.subquestions.append(sq2)
    
    # Build context for SQ2
    print("="*80)
    print("SQ2의 Previous Context 생성")
    print("="*80)
    
    context = build_context_from_previous(sq2, decomposition)
    
    print(context)
    print("\n" + "="*80)
    print("분석")
    print("="*80)
    
    if "Taquini Plan" in context:
        print("✅ Taquini Plan passage가 context에 포함되었습니다!")
    else:
        print("❌ Taquini Plan passage가 context에 없습니다!")
    
    if "Dr. Alberto Taquini" in context or "Alberto Taquini" in context:
        print("✅ 제안자 정보가 context에 포함되었습니다!")
    else:
        print("❌ 제안자 정보가 context에 없습니다!")
    
    print("\n이제 SQ2를 답변할 때, LLM은:")
    print("  1. SQ1의 답변: 'Free at initial, primary, secondary, and tertiary levels.'")
    print("  2. SQ1의 passages: Taquini Plan, Education in Argentina, Free education")
    print("  3. SQ2의 새로운 passages (if any)")
    print("를 모두 볼 수 있어서, 'Dr. Alberto Taquini'를 찾을 수 있습니다!")

if __name__ == "__main__":
    asyncio.run(test_context_building())
