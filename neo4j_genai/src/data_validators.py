from pydantic import BaseModel, PositiveInt, root_validator
from typing import List, Literal, Optional


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
    vector: Optional[EmbeddingVector] = None
    query_text: Optional[str] = None

    @root_validator(pre=True)
    def check_query(cls, values):
        vector, query_text = values.get("vector"), values.get("query_text")
        if vector and query_text:
            raise ValueError(
                "You must provide exactly one of query_vector or query_text."
            )
        return values
