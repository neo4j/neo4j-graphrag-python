from typing import List, Any, Literal, Optional
from pydantic import BaseModel, PositiveInt, Field, model_validator


class Neo4jRecord(BaseModel):
    node: Any
    score: float


class EmbeddingVector(BaseModel):
    vector: List[float]


class CreateIndexModel(BaseModel):
    name: str
    label: str
    property: str
    dimensions: int = Field(ge=1, le=2048)
    similarity_fn: Literal["euclidean", "cosine"]


class SimilaritySearchModel(BaseModel):
    index_name: str
    top_k: PositiveInt = 5
    query_vector: Optional[List[float]] = None
    query_text: Optional[str] = None

    @model_validator(mode="before")
    def check_query(cls, values):
        """
        Validates that one of either query_vector or query_text is provided exclusively.
        """
        query_vector, query_text = values.get("query_vector"), values.get("query_text")
        if not (bool(query_vector) ^ bool(query_text)):
            raise ValueError(
                "You must provide exactly one of query_vector or query_text."
            )
        return values
