import random
from typing import Any

from neo4j_graphrag.embeddings import Embedder


class CustomEmbeddings(Embedder):
    def __init__(self, model: str, dimensions: int = 10, **kwargs: Any):
        super().__init__(model, dimensions, **kwargs)

    def embed_query(self, input: str) -> list[float]:
        return [random.random() for _ in range(self.dimensions)]


llm = CustomEmbeddings("", dimensions=1024)
res = llm.embed_query("text")
print(res[:10])
