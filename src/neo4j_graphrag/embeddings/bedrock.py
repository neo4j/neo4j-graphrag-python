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
from neo4j_graphrag.utils.rate_limit import RateLimitHandler, rate_limit_handler

try:
    import boto3
except ImportError:
    boto3 = None


class BedrockEmbeddings(Embedder):
    """
    AWS Bedrock embeddings class.
    This class uses the AWS Bedrock Runtime to generate vector embeddings for text data
    using Amazon Titan Text Embeddings V2.

    Args:
        model_id (str): The Bedrock model identifier. Defaults to "amazon.titan-embed-text-v2:0".
        region_name (str, optional): AWS region name. Falls back to AWS_REGION or AWS_DEFAULT_REGION env var.
        inference_profile_id (str, optional): Inference profile ARN for cross-region inference.
            When provided, this is used instead of model_id for the invoke_model call.
        client: A pre-configured boto3 bedrock-runtime client. If provided, region_name is ignored.
        rate_limit_handler (RateLimitHandler, optional): Handler for rate limiting.
        **kwargs: Additional arguments passed to boto3.client() if client is not provided.

    Example:
        >>> from neo4j_graphrag.embeddings import BedrockEmbeddings
        >>> embedder = BedrockEmbeddings(region_name="us-east-1")
        >>> embedding = embedder.embed_query("Hello, world!")
    """

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region_name: Optional[str] = None,
        inference_profile_id: Optional[str] = None,
        client: Optional[Any] = None,
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
        Generate embeddings for a given query using Amazon Titan Text Embeddings V2.

        Args:
            text (str): The text to generate an embedding for.
            **kwargs (Any): Additional arguments passed to the Titan embedding request body.

        Returns:
            list[float]: The embedding vector.

        Raises:
            EmbeddingsGenerationError: If embedding generation fails.
        """
        # Use inference profile if provided, otherwise use model_id
        model_identifier = self.inference_profile_id or self.model_id

        # Build request body for Titan Text Embeddings V2
        request_body = {
            "inputText": text,
            **kwargs,
        }

        try:
            response = self.client.invoke_model(
                modelId=model_identifier,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            embedding: list[float] = response_body["embedding"]
            return embedding

        except Exception as e:
            # Wrap all errors in EmbeddingsGenerationError
            # The rate_limit_handler will detect rate limit patterns in the message
            # and convert to RateLimitError for retry
            # Patterns like "throttling", "rate", "429" are detected by is_rate_limit_error()
            raise EmbeddingsGenerationError(
                f"Failed to generate embedding with Bedrock: {e}"
            ) from e
