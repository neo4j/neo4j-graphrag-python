from abc import ABC, abstractmethod
from .types import EmbeddingVector


class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed query text.

        Args:
            text (str): Text to convert to vector embedding

        Returns:
            EmbeddingVector: A vector embedding.
        """
