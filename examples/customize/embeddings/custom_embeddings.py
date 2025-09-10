import random
from typing import Any

from neo4j_graphrag.embeddings import Embedder


class CustomEmbeddings(Embedder):
    def __init__(self, dimension: int = 10, **kwargs: Any):
        self.dimension = dimension

    def embed_query(
        self, input: str, dimensions: int | None = None, **kwargs: Any
    ) -> list[float]:
        v = [random.random() for _ in range(self.dimension)]
        if dimensions:
            return v[:dimensions]
        return v


llm = CustomEmbeddings(dimensions=1024)
res = llm.embed_query("text")
print(res[:10])
