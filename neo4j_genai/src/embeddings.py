from abc import ABC, abstractmethod
from typing import List
from src.data_validators import EmbeddingVector


class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed query text."""
        pass
