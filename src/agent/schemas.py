from typing import List, Literal
from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Schema for analyzing and classifying user queries."""
    category: Literal["GREETING", "INTERNAL_QUERY", "OFF_TOPIC"] = Field(
        description="The classification category of the query."
    )
    search_query: str = Field(
        description="The standalone search query optimized for vector store retrieval if category is INTERNAL_QUERY. Empty string otherwise."
    )
    reason: str = Field(
        description="Reasoning for the chosen classification category."
    )


class DocumentGrade(BaseModel):
    """Schema for a single document's relevance grade."""
    doc_index: int = Field(
        description="The index number of the document being graded."
    )
    binary_score: Literal["yes", "no"] = Field(
        description="Whether the document is strictly relevant ('yes') or not ('no') to the given query."
    )
    reason: str = Field(
        description="A concise reason explaining why the document was graded as relevant or not."
    )


class BatchGradeDecision(BaseModel):
    """Schema for the batch evaluation of multiple documents."""
    grades: List[DocumentGrade] = Field(
        description="A list of grades, one for each provided document."
    )
