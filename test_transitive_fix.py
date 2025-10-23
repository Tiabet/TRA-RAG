"""
Test if transitive dependency fix resolves the SQ3 zero-passage problem.
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from metadata_db import MetadataDB
from sequential_answering import answer_subquestions_sequential
from query_decomposition import QueryDecomposition, SubQuestion

load_dotenv()

# Initialize
client = AsyncOpenAI(
    api_key=os.getenv('ALICE_OPENAI_KEY'),
    base_url=os.getenv('ALICE_CHAT_URL')
)
db = MetadataDB('metadata_v2.db')

async def test_transitive_dependency():
    """Test the specific case that was failing."""
    
    # Manually create the decomposition from the result file
    decomposition = QueryDecomposition(
        main_query="Seven years before the opening of the Brewer Fieldhouse in Columbia, Missouri, what was a campus of the University of Missouri known as?",
        question_type="bridge",
        reasoning="This is a bridge question requiring sequential steps",
        subquestions=[
            SubQuestion(
                id="SQ1",
                question="When did the Brewer Fieldhouse in Columbia, Missouri open?",
                depends_on=[],
                reasoning="Identify the opening year of the Brewer Fieldhouse"
            ),
            SubQuestion(
                id="SQ2",
                question="What year was it seven years before [SQ1_Answer]?",
                depends_on=["SQ1"],
                reasoning="Calculate the year seven years before the opening"
            ),
            SubQuestion(
                id="SQ3",
                question="What was a campus of the University of Missouri known as in [SQ2_Answer]?",
                depends_on=["SQ2"],
                reasoning="Find what the campus was called in that year"
            )
        ]
    )
    
    print("=" * 80)
    print("Testing Transitive Dependency Fix")
    print("=" * 80)
    print(f"Main Query: {decomposition.main_query}")
    print(f"Expected Answer: University Farm")
    print()
    
    # Run sequential answering
    result = await answer_subquestions_sequential(
        client, db, decomposition,
        use_fts=True,
        apply_llm_filter_stage1a=True
    )
    
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    
    for sq in decomposition.subquestions:
        print(f"\n{sq.id}: {sq.question}")
        print(f"Answer: {sq.answer}")
        
        if hasattr(sq, 'retrieved_passages'):
            print(f"Retrieved Passages: {len(sq.retrieved_passages)}")
            for passage in sq.retrieved_passages[:3]:
                print(f"  - {passage.get('title', 'Unknown')}")
    
    print("\n" + "=" * 80)
    print("Final Answer")
    print("=" * 80)
    print(f"Predicted: {result.get('final_answer', 'N/A')}")
    print(f"Expected: University Farm")
    
    # Check success
    final_answer = result.get('final_answer', '').lower()
    if 'university farm' in final_answer:
        print("\n✅ SUCCESS! Answer contains 'University Farm'")
        return True
    else:
        print("\n❌ FAILED - Answer does not contain 'University Farm'")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_transitive_dependency())
    exit(0 if success else 1)
