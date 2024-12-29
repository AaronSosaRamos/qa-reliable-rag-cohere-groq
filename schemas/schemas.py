from typing import Any, List, TypedDict
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in 'generation' answer."""

    binary_score: str = Field(
        ...,
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class HighlightDocuments(BaseModel):
    """Return the specific part of a document used for answering the question."""

    id: List[str] = Field(
        description="List of id of docs used to answers the question"
    )

    title: List[str] = Field(
        description="List of titles used to answers the question"
    )

    source: List[str] = Field(
        description="List of sources used to answers the question"
    )

    segment: List[str] = Field(
        description="List of direct segements from used documents that answers the question. Don't add additional symbols, just return the text from the document"
    )

class GraphState(TypedDict):
    urls: str
    question: str

    retriever: Any
    docs_to_use: Any
    generation: str
    is_grounded: str
    lookup_response: HighlightDocuments