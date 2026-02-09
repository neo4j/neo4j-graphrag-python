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

# built-in dependencies
from typing import Any, Optional

# project dependencies
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
    async_rate_limit_handler,
    rate_limit_handler,
)

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

DEFAULT_EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_EMBEDDING_DIM = 768


class GeminiEmbedder(Embedder):
    """Embedder that uses Google's Gemini API via the google.genai SDK.

    Args:
        model: Embedding model name. Defaults to "text-embedding-004".
        embedding_dim: Output dimensionality. Defaults to 768.
        rate_limit_handler: Optional rate limit handler.
        **kwargs: Arguments passed to the genai.Client.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ) -> None:
        if genai is None or types is None:
            raise ImportError(
                "Could not import google-genai python client. "
                'Please install it with `pip install "neo4j-graphrag[google-genai]"`.'
            )
        super().__init__(rate_limit_handler)
        self.model = model
        self.embedding_dim = embedding_dim
        self.client = genai.Client(**kwargs)

    @rate_limit_handler
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=[text],
                config=types.EmbedContentConfig(
                    output_dimensionality=self.embedding_dim
                ),
                **kwargs,
            )
            if not result.embeddings or not result.embeddings[0].values:
                raise ValueError("No embeddings returned from Gemini API")
            return list(result.embeddings[0].values)
        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with Gemini: {e}"
            ) from e

    @async_rate_limit_handler
    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        try:
            result = await self.client.aio.models.embed_content(
                model=self.model,
                contents=[text],
                config=types.EmbedContentConfig(
                    output_dimensionality=self.embedding_dim
                ),
                **kwargs,
            )
            if not result.embeddings or not result.embeddings[0].values:
                raise ValueError("No embeddings returned from Gemini API")
            return list(result.embeddings[0].values)
        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with Gemini: {e}"
            ) from e
