import random
from typing import Any

from neo4j_graphrag.embeddings import Embedder


class CustomEmbeddings(Embedder):
    def __init__(self, dimension: int = 10, **kwargs: Any):
        self.dimension = dimension

    def embed_query(self, input: str, **kwargs) -> list[float]:
        return [random.random() for _ in range(self.dimension)]


llm = CustomEmbeddings(dimensions=1024)
res = llm.embed_query("text")
print(res[:10])
