from pydantic import BaseModel, PositiveInt, root_validator
from src.embeddings import EmbeddingVector
from typing import List, Literal, Optional


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
        if bool(vector) == bool(query_text):
            raise ValueError("You must provide exactly one of query_vector or query_text.")
        return values
