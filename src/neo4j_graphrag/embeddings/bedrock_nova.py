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
from typing import Any, Literal, Optional

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.utils.rate_limit import RateLimitHandler, rate_limit_handler

try:
    import boto3
except ImportError:
    boto3 = None


class BedrockNovaEmbeddings(Embedder):
    """
    AWS Bedrock Nova Multimodal Embeddings class.
    This class uses the AWS Bedrock Runtime to generate vector embeddings for text data
    using Amazon Nova Multimodal Embeddings.

    Nova uses a different request/response format than Titan embeddings and supports
    multimodal inputs (text, image, audio, video), configurable embedding dimensions,
    and purpose-optimized embeddings.

    Args:
        model_id (str): The Bedrock model identifier.
            Defaults to "amazon.nova-2-multimodal-embeddings-v1:0".
        region_name (str, optional): AWS region name. Falls back to AWS_REGION or
            AWS_DEFAULT_REGION env var.
        inference_profile_id (str, optional): Inference profile ARN for cross-region
            inference. When provided, this is used instead of model_id for the
            invoke_model call.
        client: A pre-configured boto3 bedrock-runtime client. If provided,
            region_name is ignored.
        embedding_dimension (int): The size of the embedding vector.
            Allowed values: 256, 384, 1024, 3072. Defaults to 1024.
        embedding_purpose (str): The intended use of the embedding.
            Defaults to "GENERIC_INDEX".
        truncation_mode (str): How to handle text that exceeds the model's
            token limit. Allowed values: "START", "END", "NONE".
            Defaults to "END".
        rate_limit_handler (RateLimitHandler, optional): Handler for rate limiting.
        **kwargs: Additional arguments passed to boto3.client() if client is not provided.

    Example:
        >>> from neo4j_graphrag.embeddings import BedrockNovaEmbeddings
        >>> embedder = BedrockNovaEmbeddings(region_name="us-east-1")
        >>> embedding = embedder.embed_query("Hello, world!")

    Example with retrieval-optimized embeddings:
        >>> embedder = BedrockNovaEmbeddings(
        ...     region_name="us-east-1",
        ...     embedding_purpose="TEXT_RETRIEVAL",
        ...     embedding_dimension=3072,
        ... )
    """

    def __init__(
        self,
        model_id: str = "amazon.nova-2-multimodal-embeddings-v1:0",
        region_name: Optional[str] = None,
        inference_profile_id: Optional[str] = None,
        client: Optional[Any] = None,
        embedding_dimension: Literal[256, 384, 1024, 3072] = 1024,
        embedding_purpose: str = "GENERIC_INDEX",
        truncation_mode: Literal["START", "END", "NONE"] = "END",
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ) -> None:
        if boto3 is None:
            raise ImportError(
                """Could not import boto3 python client.
                Please install it with `pip install "neo4j-graphrag[bedrock]"`."""
            )
        super().__init__(rate_limit_handler)
        self.model_id = model_id
        self.inference_profile_id = inference_profile_id
        self.embedding_dimension = embedding_dimension
        self.embedding_purpose = embedding_purpose
        self.truncation_mode = truncation_mode

        if client is not None:
            self.client = client
        else:
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=region_name,
                **kwargs,
            )

    @rate_limit_handler
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """
        Generate embeddings for a given query using Amazon Nova Multimodal Embeddings.

        Args:
            text (str): The text to generate an embedding for.
            **kwargs (Any): Additional arguments merged into singleEmbeddingParams.

        Returns:
            list[float]: The embedding vector.

        Raises:
            EmbeddingsGenerationError: If embedding generation fails.
        """
        model_identifier = self.inference_profile_id or self.model_id

        request_body: dict[str, Any] = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": self.embedding_purpose,
                "embeddingDimension": self.embedding_dimension,
                "text": {
                    "truncationMode": self.truncation_mode,
                    "value": text,
                },
                **kwargs,
            },
        }

        try:
            response = self.client.invoke_model(
                modelId=model_identifier,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            embedding: list[float] = response_body["embeddings"][0]["embedding"]
            return embedding

        except Exception as e:
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with Bedrock Nova: {e}"
            ) from e
