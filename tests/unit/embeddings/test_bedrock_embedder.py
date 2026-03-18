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
import io
import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from tenacity import RetryError

from neo4j_graphrag.embeddings.bedrock import BedrockEmbeddings
from neo4j_graphrag.exceptions import EmbeddingsGenerationError


@patch("neo4j_graphrag.embeddings.bedrock.boto3", None)
def test_bedrock_embedder_missing_dependency() -> None:
    with pytest.raises(ImportError):
        BedrockEmbeddings()


@patch("neo4j_graphrag.embeddings.bedrock.boto3")
def test_bedrock_embedder_happy_path(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    # Mock the response body
    response_body = {"embedding": [1.0, 2.0, 3.0]}
    mock_client.invoke_model.return_value = {
        "body": io.BytesIO(json.dumps(response_body).encode("utf-8"))
    }

    embedder = BedrockEmbeddings(region_name="us-east-1")
    res = embedder.embed_query("my text")

    assert isinstance(res, list)
    assert res == [1.0, 2.0, 3.0]

    # Verify boto3.client was called with correct parameters
    mock_boto3.client.assert_called_once_with(
        "bedrock-runtime",
        region_name="us-east-1",
    )


@patch("neo4j_graphrag.embeddings.bedrock.boto3")
def test_bedrock_embedder_default_model(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    response_body = {"embedding": [1.0, 2.0]}
    mock_client.invoke_model.return_value = {
        "body": io.BytesIO(json.dumps(response_body).encode("utf-8"))
    }

    embedder = BedrockEmbeddings()
    embedder.embed_query("test")

    # Verify the default model was used
    call_args = mock_client.invoke_model.call_args
    assert call_args.kwargs["modelId"] == "amazon.titan-embed-text-v2:0"


@patch("neo4j_graphrag.embeddings.bedrock.boto3")
def test_bedrock_embedder_custom_model(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    response_body = {"embedding": [1.0, 2.0]}
    mock_client.invoke_model.return_value = {
        "body": io.BytesIO(json.dumps(response_body).encode("utf-8"))
    }

    embedder = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
    embedder.embed_query("test")

    call_args = mock_client.invoke_model.call_args
    assert call_args.kwargs["modelId"] == "amazon.titan-embed-text-v1"


@patch("neo4j_graphrag.embeddings.bedrock.boto3")
def test_bedrock_embedder_inference_profile(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    response_body = {"embedding": [1.0, 2.0]}
    mock_client.invoke_model.return_value = {
        "body": io.BytesIO(json.dumps(response_body).encode("utf-8"))
    }

    inference_profile_arn = (
        "arn:aws:bedrock:us-east-1:123456789:inference-profile/my-profile"
    )
    embedder = BedrockEmbeddings(inference_profile_id=inference_profile_arn)
    embedder.embed_query("test")

    # Verify inference profile was used instead of model_id
    call_args = mock_client.invoke_model.call_args
    assert call_args.kwargs["modelId"] == inference_profile_arn


@patch("neo4j_graphrag.embeddings.bedrock.boto3")
def test_bedrock_embedder_with_preconfigured_client(mock_boto3: Mock) -> None:
    # Create a mock pre-configured client
    mock_client = MagicMock()
    response_body = {"embedding": [1.0, 2.0]}
    mock_client.invoke_model.return_value = {
        "body": io.BytesIO(json.dumps(response_body).encode("utf-8"))
    }

    embedder = BedrockEmbeddings(client=mock_client)
    res = embedder.embed_query("test")

    assert res == [1.0, 2.0]
    # Verify boto3.client was NOT called since we provided a client
    mock_boto3.client.assert_not_called()


@patch("neo4j_graphrag.embeddings.bedrock.boto3")
def test_bedrock_embedder_request_body_format(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    response_body = {"embedding": [1.0, 2.0]}
    mock_client.invoke_model.return_value = {
        "body": io.BytesIO(json.dumps(response_body).encode("utf-8"))
    }

    embedder = BedrockEmbeddings()
    embedder.embed_query("Hello, world!")

    call_args = mock_client.invoke_model.call_args
    request_body = json.loads(call_args.kwargs["body"])

    assert request_body["inputText"] == "Hello, world!"
    assert call_args.kwargs["contentType"] == "application/json"
    assert call_args.kwargs["accept"] == "application/json"


@patch("neo4j_graphrag.embeddings.bedrock.boto3")
def test_bedrock_embedder_non_retryable_error(mock_boto3: Mock) -> None:
    """Test that non-retryable errors fail immediately without retries."""
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.invoke_model.side_effect = Exception("ValidationException: Invalid input")

    embedder = BedrockEmbeddings()

    with pytest.raises(
        EmbeddingsGenerationError, match="Failed to generate embedding with Bedrock"
    ):
        embedder.embed_query("my text")

    # Verify the API was called only once (no retries for non-rate-limit errors)
    assert mock_client.invoke_model.call_count == 1


@patch("neo4j_graphrag.embeddings.bedrock.boto3")
def test_bedrock_embedder_rate_limit_error_retries(mock_boto3: Mock) -> None:
    """Test that rate limit errors are retried the expected number of times."""
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    # Use "rate limit" pattern which is detected by is_rate_limit_error()
    mock_client.invoke_model.side_effect = [
        Exception("rate limit exceeded"),
        Exception("rate limit exceeded"),
        Exception("rate limit exceeded"),
    ]

    embedder = BedrockEmbeddings()

    # After exhausting retries, tenacity raises RetryError
    with pytest.raises(RetryError):
        embedder.embed_query("my text")

    # Verify the API was called 3 times (default max_attempts for RetryRateLimitHandler)
    assert mock_client.invoke_model.call_count == 3


@patch("neo4j_graphrag.embeddings.bedrock.boto3")
def test_bedrock_embedder_rate_limit_eventual_success(mock_boto3: Mock) -> None:
    """Test that rate limit errors eventually succeed after retries."""
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    # First two calls fail with rate limit, third succeeds
    response_body = {"embedding": [1.0, 2.0, 3.0]}
    mock_response = {"body": io.BytesIO(json.dumps(response_body).encode("utf-8"))}

    mock_client.invoke_model.side_effect = [
        Exception("rate limit exceeded"),
        Exception("rate limit exceeded"),
        mock_response,
    ]

    embedder = BedrockEmbeddings()
    result = embedder.embed_query("my text")

    # Verify successful result
    assert result == [1.0, 2.0, 3.0]
    # Verify the API was called 3 times before succeeding
    assert mock_client.invoke_model.call_count == 3
