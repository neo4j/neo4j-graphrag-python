from neo4j_genai.embedder import Embedder


class SentenceTransformerEmbeddings(Embedder):
    def __init__(self, model="all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            )

        self.model = SentenceTransformer(model)

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text])
