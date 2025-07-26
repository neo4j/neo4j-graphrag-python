from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
import boto3
import json
import time
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError


class BedrockEmbeddings(Embedder):
    """
    Embedder implementation using Amazon Bedrock's Titan Text Embedding model.

    This class integrates with AWS Bedrock via `boto3` and uses the Titan Embedding
    model (`amazon.titan-embed-text-v2:0`) to generate 1536-dimensional vector
    representations for input text.

    Example:
        >>> embedder = BedrockEmbeddings()
        >>> embedding = embedder.embed_query("Neo4j integrates well with Bedrock.")
        >>> len(embedding)
        1536

    Notes:
        - Embeddings returned are 1536-dimensional vectors.
        - A short sleep delay is applied to avoid throttling.
        - This class uses the default AWS credentials chain supported by `boto3`.

    AWS Authentication:
        The following authentication methods are supported through boto3:

        - Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (if needed)
        - AWS credentials/config files (e.g., `~/.aws/credentials`)
        - IAM roles (if running on EC2, Lambda, SageMaker, etc.)
        - AWS CLI named profile via `AWS_PROFILE` environment variable
    """

    def __init__(
        self,
        model_id: str = 'amazon.titan-embed-text-v2:0',
        region: str = 'us-east-1'
    ):
        """
        Initialize the BedrockEmbeddings instance.

        Args:
            model_id (str): Identifier for the Bedrock Titan embedding model.
                            Default is 'amazon.titan-embed-text-v2:0'.
            region (str): AWS region where the Bedrock service is hosted.
                          Default is 'us-east-1'.
        """
        self.model_id = model_id
        self.bedrock = boto3.client('bedrock-runtime', region_name=region)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate a vector embedding for the input text using Amazon Bedrock.

        Args:
            text (str): The input text string to be embedded.

        Returns:
            List[float]: A 1536-dimensional list representing the text embedding.

        Raises:
            EmbeddingsGenerationError: If an error occurs during the embedding process.
        """
        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps({"inputText": text})
            )
            body = json.loads(response['body'].read())
            time.sleep(0.05)  # To prevent throttling
            return body['embedding']
        except Exception as e:
            raise EmbeddingsGenerationError(f"Issue Generating Embeddings: {e}")
