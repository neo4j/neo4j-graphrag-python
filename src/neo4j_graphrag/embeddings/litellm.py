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

from typing import Any, Optional

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
    async_rate_limit_handler,
    rate_limit_handler,
)


class LiteLLMEmbeddings(Embedder):
    """Embeddings via LiteLLM AI gateway.

    LiteLLM provides a unified interface to 100+ embedding providers using the
    OpenAI embedding format.

    Args:
        model (str): The embedding model identifier in LiteLLM format
            (e.g. "text-embedding-ada-002", "cohere/embed-english-v3.0").
        rate_limit_handler (Optional[RateLimitHandler]): A rate limit handler
            to manage API rate limits. Defaults to None.
        **kwargs (Any): Arguments passed to ``litellm.embedding``
            (e.g. ``api_key``, ``api_base``).

    Raises:
        EmbeddingsGenerationError: If there's an error generating embeddings.

    Example:

    .. code-block:: python

        from neo4j_graphrag.embeddings import LiteLLMEmbeddings

        embedder = LiteLLMEmbeddings(model="text-embedding-ada-002", api_key="...")
        vector = embedder.embed_query("Some text")
    """

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ) -> None:
        try:
            import litellm
        except ImportError:
            raise ImportError(
                """Could not import litellm python client.
                Please install it with `pip install "neo4j-graphrag[litellm]"`."""
            )
        super().__init__(rate_limit_handler)
        self.litellm = litellm
        self.model = model
        self.kwargs = kwargs

    @rate_limit_handler
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Generate embeddings for a given query using LiteLLM.

        Args:
            text (str): The text to generate an embedding for.
            **kwargs (Any): Additional arguments passed to ``litellm.embedding``.
        """
        try:
            response = self.litellm.embedding(
                model=self.model,
                input=[text],
                **self.kwargs,
                **kwargs,
            )
            embedding: list[float] = response.data[0]["embedding"]
            return embedding
        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with LiteLLM: {e}"
            ) from e

    @async_rate_limit_handler
    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Asynchronously generate embeddings for a given query using LiteLLM.

        Args:
            text (str): The text to generate an embedding for.
            **kwargs (Any): Additional arguments passed to ``litellm.aembedding``.
        """
        try:
            response = await self.litellm.aembedding(
                model=self.model,
                input=[text],
                **self.kwargs,
                **kwargs,
            )
            embedding: list[float] = response.data[0]["embedding"]
            return embedding
        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with LiteLLM: {e}"
            ) from e
