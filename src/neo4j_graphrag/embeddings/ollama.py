#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

from typing import Any

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError


class OllamaEmbeddings(Embedder):
    """
    Ollama embeddings class.
    This class uses the ollama Python client to generate vector embeddings for text data.

    Args:
        model (str): The name of the Mistral AI text embedding model to use. Defaults to "mistral-embed".
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "Could not import ollama python client. "
                "Please install it with `pip install ollama`."
            )
        self.model = model
        self.client = ollama.Client(**kwargs)

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """
        Generate embeddings for a given query using an Ollama text embedding model.

        Args:
            text (str): The text to generate an embedding for.
            **kwargs (Any): Additional keyword arguments to pass to the Ollama client.
        """
        embeddings_response = self.client.embed(
            model=self.model,
            input=text,
            **kwargs,
        )

        if embeddings_response is None or embeddings_response.embeddings is None:
            raise EmbeddingsGenerationError("Failed to retrieve embeddings.")

        embedding = embeddings_response.embeddings
        if not isinstance(embedding, list):
            raise EmbeddingsGenerationError("Embedding is not a list of floats.")

        return embedding
