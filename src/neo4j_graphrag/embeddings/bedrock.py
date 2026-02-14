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

import json
from typing import Any, Optional

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
    async_rate_limit_handler,
    rate_limit_handler,
)

try:
    import boto3
except ImportError:
    boto3 = None

DEFAULT_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_DIMENSIONS = 1024


class BedrockEmbeddings(Embedder):
    """Embedder that uses Amazon Bedrock's embedding models via the boto3 SDK.

    Supports Amazon Titan Embed and Cohere Embed models available through Bedrock.

    Args:
        model_id: Bedrock model ID. Defaults to "amazon.titan-embed-text-v2:0".
        dimensions: Output embedding dimensionality. Defaults to 1024.
        normalize: Whether to normalize the embedding vector. Defaults to True.
        region_name: AWS region. Defaults to boto3 session default.
        rate_limit_handler: Optional rate limit handler.
        **kwargs: Arguments passed to ``boto3.client("bedrock-runtime", ...)``.

    Example:

    .. code-block:: python

        from neo4j_graphrag.embeddings import BedrockEmbeddings

        embedder = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            dimensions=1024,
            region_name="us-east-1",
        )
        vector = embedder.embed_query("my question")
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        dimensions: int = DEFAULT_DIMENSIONS,
        normalize: bool = True,
        region_name: Optional[str] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ) -> None:
        if boto3 is None:
            raise ImportError(
                "Could not import boto3 python client. "
                'Please install it with `pip install "neo4j-graphrag[bedrock]"`.'
            )
        super().__init__(rate_limit_handler)
        self.model_id = model_id
        self.dimensions = dimensions
        self.normalize = normalize
        client_kwargs: dict[str, Any] = {**kwargs}
        if region_name:
            client_kwargs["region_name"] = region_name
        self.client = boto3.client("bedrock-runtime", **client_kwargs)

    @rate_limit_handler
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        try:
            body = json.dumps(
                {
                    "inputText": text,
                    "dimensions": self.dimensions,
                    "normalize": self.normalize,
                }
            )
            response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response["body"].read())
            embedding = response_body.get("embedding")
            if not embedding:
                raise ValueError("No embedding returned from Bedrock API")
            return list(embedding)
        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with Bedrock: {e}"
            ) from e

    @async_rate_limit_handler
    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        # boto3 does not have native async support; run synchronously
        try:
            body = json.dumps(
                {
                    "inputText": text,
                    "dimensions": self.dimensions,
                    "normalize": self.normalize,
                }
            )
            response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response["body"].read())
            embedding = response_body.get("embedding")
            if not embedding:
                raise ValueError("No embedding returned from Bedrock API")
            return list(embedding)
        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with Bedrock: {e}"
            ) from e
