from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel


class EmbeddingVector(BaseModel):
    vector: List[float]


class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed query text."""
        pass
