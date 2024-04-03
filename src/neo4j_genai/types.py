from typing import List, Any, Literal, Optional
from pydantic import BaseModel, PositiveInt, model_validator, field_validator
from neo4j import Driver


class Neo4jRecord(BaseModel):
    node: Any
    score: float


class EmbeddingVector(BaseModel):
    vector: List[float]


class IndexModel(BaseModel):
    driver: Any

    @field_validator("driver")
    def check_driver_is_valid(cls, v):
        if not isinstance(v, Driver):
            raise ValueError("driver must be an instance of neo4j.Driver")
        return v


class VectorIndexModel(IndexModel):
    name: str
    label: str
    property: str
    dimensions: PositiveInt
    similarity_fn: Literal["euclidean", "cosine"]


class FulltextIndexModel(IndexModel):
    name: str
    label: str
    node_properties: List[str]

    @field_validator("node_properties")
    def check_node_properties_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("node_properties cannot be an empty list")
        return v


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
