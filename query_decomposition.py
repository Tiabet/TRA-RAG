"""
Query Decomposition Module
============================
Decomposes multi-hop questions into sequential sub-questions.

Supports:
- Bridge questions: Sequential dependency chain (SQ1 → SQ2 → SQ3)
- Comparison questions: Parallel retrieval + synthesis (SQ1 + SQ2 → Compare)
"""

import json
from typing import Dict, List, Optional
from openai import AsyncOpenAI

from Prompt.query_decomposition_prompt import QUERY_DECOMPOSITION_PROMPT


class SubQuestion:
    """Represents a single sub-question in the decomposition"""
    
    def __init__(
        self,
        id: str,
        question: str,
        depends_on: List[str],
        reasoning: str
    ):
        self.id = id
        self.question = question
        self.depends_on = depends_on  # List of SQ IDs this depends on
        self.reasoning = reasoning
        
        # To be filled during execution
        self.answer = None
        self.retrieved_passages = []
        self.retrieval_info = {}
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'question': self.question,
            'depends_on': self.depends_on,
            'reasoning': self.reasoning,
            'answer': self.answer,
            'retrieved_passages': self.retrieved_passages,
            'retrieval_info': self.retrieval_info
        }
    
    def __repr__(self):
        deps = f" (depends on: {', '.join(self.depends_on)})" if self.depends_on else ""
        return f"SubQuestion({self.id}: {self.question}{deps})"


class QueryDecomposition:
    """Result of query decomposition"""
    
    def __init__(
        self,
        main_query: str,
        question_type: str,
        reasoning: str,
        subquestions: List[SubQuestion]
    ):
        self.main_query = main_query
        self.question_type = question_type  # "bridge" or "comparison"
        self.reasoning = reasoning
        self.subquestions = subquestions
        
    def get_subquestion(self, sq_id: str) -> Optional[SubQuestion]:
        """Get a specific sub-question by ID"""
        for sq in self.subquestions:
            if sq.id == sq_id:
                return sq
        return None
    
    def get_independent_subquestions(self) -> List[SubQuestion]:
        """Get sub-questions with no dependencies (can be run in parallel)"""
        return [sq for sq in self.subquestions if not sq.depends_on]
    
    def get_next_subquestion(self) -> Optional[SubQuestion]:
        """Get the next unanswered sub-question whose dependencies are satisfied"""
        for sq in self.subquestions:
            # Skip if already answered
            if sq.answer is not None:
                continue
            
            # Check if all dependencies are satisfied
            deps_satisfied = all(
                self.get_subquestion(dep_id).answer is not None
                for dep_id in sq.depends_on
            )
            
            if deps_satisfied:
                return sq
        
        return None
    
    def is_complete(self) -> bool:
        """Check if all sub-questions have been answered"""
        return all(sq.answer is not None for sq in self.subquestions)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'main_query': self.main_query,
            'question_type': self.question_type,
            'reasoning': self.reasoning,
            'subquestions': [sq.to_dict() for sq in self.subquestions]
        }
    
    def __repr__(self):
        return (f"QueryDecomposition(type={self.question_type}, "
                f"{len(self.subquestions)} sub-questions)")


async def decompose_query(
    client: AsyncOpenAI,
    query: str,
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.1
) -> Dict:
    """
    Decompose a multi-hop question into sequential sub-questions.
    
    Args:
        client: AsyncOpenAI client
        query: The main question to decompose
        model: LLM model to use
        temperature: Temperature for generation (low for consistency)
        
    Returns:
        Dict with 'success', 'decomposition' (QueryDecomposition object), and optional 'error'
    """
    try:
        # Format prompt
        formatted_prompt = QUERY_DECOMPOSITION_PROMPT.replace("__QUESTION__", query)
        
        # Call LLM
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at decomposing complex questions into simpler sub-questions."
                },
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ],
            temperature=temperature,
            max_tokens=2048
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
        
        # Validate required fields
        if 'question_type' not in result:
            return {
                'success': False,
                'error': "Missing 'question_type' in decomposition result"
            }
        
        if 'subquestions' not in result or not result['subquestions']:
            return {
                'success': False,
                'error': "Missing or empty 'subquestions' in decomposition result"
            }
        
        # Create SubQuestion objects
        subquestions = []
        for sq_data in result['subquestions']:
            sq = SubQuestion(
                id=sq_data['id'],
                question=sq_data['question'],
                depends_on=sq_data.get('depends_on', []),
                reasoning=sq_data.get('reasoning', '')
            )
            subquestions.append(sq)
        
        # Create QueryDecomposition object
        decomposition = QueryDecomposition(
            main_query=query,
            question_type=result['question_type'],
            reasoning=result.get('reasoning', ''),
            subquestions=subquestions
        )
        
        return {
            'success': True,
            'decomposition': decomposition,
            'raw_response': result
        }
        
    except json.JSONDecodeError as e:
        return {
            'success': False,
            'error': f"Failed to parse JSON: {str(e)}",
            'raw_response': result_text if 'result_text' in locals() else None
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Decomposition failed: {str(e)}"
        }


def substitute_answers(question: str, answered_subquestions: List[SubQuestion]) -> str:
    """
    Substitute [SQ{N}_Answer] placeholders with actual answers.
    
    Args:
        question: Question with placeholders like [SQ1_Answer], [SQ2_Answer]
        answered_subquestions: List of SubQuestion objects with answers
        
    Returns:
        Question with placeholders replaced by actual answers
    """
    result = question
    
    for sq in answered_subquestions:
        if sq.answer:
            placeholder = f"[{sq.id}_Answer]"
            result = result.replace(placeholder, sq.answer)
    
    return result


def build_context_from_previous(
    current_sq: SubQuestion,
    decomposition: QueryDecomposition
) -> str:
    """
    Build context string from previous answered sub-questions.
    Includes both answers AND retrieved passages to provide full context.
    
    Args:
        current_sq: Current sub-question being processed
        decomposition: QueryDecomposition with previous answers
        
    Returns:
        Context string with previous Q&A pairs and passages
    """
    context_parts = []
    
    # Get all dependencies
    for dep_id in current_sq.depends_on:
        dep_sq = decomposition.get_subquestion(dep_id)
        if dep_sq and dep_sq.answer:
            context_parts.append(f"{dep_sq.id}: {dep_sq.question}")
            context_parts.append(f"Answer: {dep_sq.answer}")
            
            # Add retrieved passages from previous SQ (CRITICAL for dependent questions!)
            if hasattr(dep_sq, 'retrieved_passages') and dep_sq.retrieved_passages:
                context_parts.append(f"\nRetrieved Passages for {dep_sq.id}:")
                for i, passage in enumerate(dep_sq.retrieved_passages[:5], 1):  # Top 5 passages
                    title = passage.get('title', 'Unknown')
                    context_parts.append(f"  [{i}] {title}")
                    
                    # Add FULL metadata (no truncation)
                    metadata = passage.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            import json
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    
                    # Show ALL metadata fields fully
                    for key in ['description', 'main_entity', 'attributes', 'events']:
                        if key in metadata and metadata[key]:
                            value = str(metadata[key])  # FULL value, no truncation!
                            context_parts.append(f"      {key}: {value}")
            
            context_parts.append("")  # Empty line for readability
    
    if context_parts:
        return "Previous Sub-Questions and Answers:\n" + "\n".join(context_parts)
    else:
        return ""


def get_execution_order(decomposition: QueryDecomposition) -> List[List[str]]:
    """
    Get execution order for sub-questions, grouping independent ones.
    
    Returns:
        List of batches, where each batch contains SQ IDs that can be executed in parallel
        
    Example:
        Bridge: [["SQ1"], ["SQ2"], ["SQ3"]]  # Sequential
        Comparison: [["SQ1", "SQ2"], ["SQ3"]]  # SQ1+SQ2 parallel, then SQ3
    """
    batches = []
    answered = set()
    
    while len(answered) < len(decomposition.subquestions):
        # Find all SQs whose dependencies are satisfied
        ready = []
        for sq in decomposition.subquestions:
            if sq.id in answered:
                continue
            
            # Check if all dependencies are answered
            if all(dep_id in answered for dep_id in sq.depends_on):
                ready.append(sq.id)
        
        if not ready:
            # Should not happen if decomposition is valid
            raise ValueError("Circular dependency or invalid decomposition")
        
        batches.append(ready)
        answered.update(ready)
    
    return batches


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def test_decomposition():
        """Test query decomposition with example questions"""
        
        client = AsyncOpenAI(
            api_key=os.getenv('ALICE_OPENAI_KEY'),
            base_url=os.getenv('ALICE_CHAT_URL')
        )
        
        # Test cases
        test_queries = [
            # Bridge question
            "In which international tournament did the 23rd overall pick of the 2015 NHL Entry Draft help the United States national junior team win a bronze medal, and in what city was it held?",
            
            # Comparison question
            "Are both Stephen R. Donaldson and Michael Moorcock science fiction writers?",
            
            # Another bridge
            "Who proposed the plan for free education in Argentina?",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"Test {i}: {query}")
            print('='*80)
            
            result = await decompose_query(client, query)
            
            if result['success']:
                decomp = result['decomposition']
                print(f"\nQuestion Type: {decomp.question_type}")
                print(f"Reasoning: {decomp.reasoning}")
                print(f"\nSub-Questions ({len(decomp.subquestions)}):")
                
                for sq in decomp.subquestions:
                    deps = f" [depends on: {', '.join(sq.depends_on)}]" if sq.depends_on else ""
                    print(f"\n  {sq.id}: {sq.question}{deps}")
                    print(f"  → Reasoning: {sq.reasoning}")
                
                # Show execution order
                execution_order = get_execution_order(decomp)
                print(f"\nExecution Order:")
                for batch_idx, batch in enumerate(execution_order, 1):
                    parallel = " (parallel)" if len(batch) > 1 else ""
                    print(f"  Batch {batch_idx}: {', '.join(batch)}{parallel}")
            else:
                print(f"\n❌ Error: {result['error']}")
                if result.get('raw_response'):
                    print(f"Raw response: {result['raw_response']}")
    
    # Run test
    asyncio.run(test_decomposition())
