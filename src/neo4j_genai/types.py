from pydantic import BaseModel, PositiveInt, root_validator
from typing import List, Literal, Optional


# class DatabaseQueryResult:
#     node
#     score: float
#     id: str


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
    query_vector: Optional[EmbeddingVector] = None
    query_text: Optional[str] = None

    @root_validator(pre=True)
    def check_query(cls, values):
        query_vector, query_text = values.get("query_vector"), values.get("query_text")
        if bool(query_vector) ^ bool(query_text):
            raise ValueError(
                "You must provide exactly one of query_vector or query_text."
            )
        return values
