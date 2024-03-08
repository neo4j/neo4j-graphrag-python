from typing import List, Literal, Optional, Any, Dict
from pydantic import BaseModel, PositiveInt, model_validator


class Neo4jRecord(BaseModel):
    node: Any
    score: float


class EmbeddingVector(BaseModel):
    vector: List[float]


class CreateIndexModel(BaseModel):
    name: str
    label: str
    property: str
    dimensions: PositiveInt
    similarity_fn: Literal["euclidean", "cosine"]


class SimilaritySearchModel(BaseModel):
    index_name: str
    top_k: PositiveInt = 5
    query_vector: Optional[List[float]] = None
    query_text: Optional[str] = None

    @model_validator(mode="before")
    def check_only_either_vector_or_text(cls, values):
        """
        Validates that one of either query_vector or query_text is provided exclusively.
        """
        query_vector, query_text = values.get("query_vector"), values.get("query_text")
        if not (bool(query_vector) ^ bool(query_text)):
            raise ValueError(
                "You must provide exactly one of query_vector or query_text."
            )
        return values


class CustomSimilaritySearchModel(SimilaritySearchModel):
    custom_retrieval_query: str
    custom_params: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    def combine_custom_params(cls, values):
        """
        Combine custom_params dict into the main model's fields.
        """
        custom_params = values.pop("custom_params", None) or {}
        for key, value in custom_params.items():
            if key not in values:
                values[key] = value
        return values
