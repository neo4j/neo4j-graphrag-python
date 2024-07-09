from __future__ import annotations

from typing import Any

from neo4j_genai.embedder import Embedder


class OpenAIEmbeddings(Embedder):
    def __init__(self, model: str = "text-embedding-ada-002") -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Could not import openai python client. "
                "Please install it with `pip install openai`."
            )

        self.openai_model = openai.OpenAI()
        self.model = model

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        response = self.openai_model.embeddings.create(
            input=text, model=self.model, **kwargs
        )
        return response.data[0].embedding
