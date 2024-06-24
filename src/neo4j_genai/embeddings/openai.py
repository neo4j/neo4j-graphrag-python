from typing import Any
from neo4j_genai.embedder import Embedder
from typing import List


class OpenAIEmbeddings(Embedder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Could not import openai python client. "
                "Please install it with `pip install openai`."
            )

        self.model = openai.OpenAI(*args, **kwargs)

    def embed_query(
        self, text: str, model: str = "text-embedding-ada-002", **kwargs: Any
    ) -> List[float]:
        response = self.model.embeddings.create(input=text, model=model, **kwargs)
        return response.data[0].embedding
