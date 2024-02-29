from abc import ABC, abstractmethod
from typing import List


class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
