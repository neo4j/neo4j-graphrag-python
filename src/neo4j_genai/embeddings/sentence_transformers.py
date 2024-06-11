from neo4j_genai.embedder import Embedder
from typing import Any
import torch
import numpy as np


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
        result = self.model.encode([text])
        if isinstance(result, torch.Tensor) or isinstance(result, np.ndarray):
            return result.flatten().tolist()
        elif isinstance(result, list) and all(
            isinstance(x, torch.Tensor) for x in result
        ):
            return [item for tensor in result for item in tensor.flatten().tolist()]
        else:
            raise ValueError("Unexpected return type from model encoding")
