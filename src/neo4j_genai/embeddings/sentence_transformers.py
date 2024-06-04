from neo4j_genai.embedder import Embedder
from typing import Any


class SentenceTransformerEmbeddings(Embedder):
    def __init__(
        self, model: str = "all-MiniLM-L6-v2", *args: Any, **kwargs: Any
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            )

        self.model = SentenceTransformer(model, *args, **kwargs)

    def embed_query(self, text: str) -> Any:
        return self.model.encode([text]).flatten().tolist()
