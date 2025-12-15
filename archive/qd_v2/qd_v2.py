"""QD_v2 schema helpers.

This file defines a serialization-friendly format that extends the existing
QueryDecomposition output with passage selection results per sub-question.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class SubQuestionV2:
    id: str
    question: str
    depends_on: List[str]
    reasoning: str = ""

    answer: Optional[str] = None

    # Raw retrieval outputs (as produced by the pipeline)
    retrieved_passages: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_info: Dict[str, Any] = field(default_factory=dict)

    # Filtered passages judged necessary for answering this SQ
    selected_passages: List[Dict[str, Any]] = field(default_factory=list)
    selection_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryDecompositionV2:
    main_query: str
    question_type: str
    reasoning: str
    subquestions: List[SubQuestionV2]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "main_query": self.main_query,
            "question_type": self.question_type,
            "reasoning": self.reasoning,
            "subquestions": [sq.to_dict() for sq in self.subquestions],
        }
