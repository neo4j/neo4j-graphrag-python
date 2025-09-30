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
from unittest.mock import MagicMock, Mock, patch

import pytest
from tenacity import RetryError
from neo4j_graphrag.embeddings import MistralAIEmbeddings
from neo4j_graphrag.exceptions import EmbeddingsGenerationError


@patch("neo4j_graphrag.embeddings.mistral.Mistral", None)
def test_mistralai_embedder_missing_dependency() -> None:
    with pytest.raises(ImportError):
        MistralAIEmbeddings()


@patch("neo4j_graphrag.embeddings.mistral.Mistral")
def test_mistralai_embedder_happy_path(mock_mistralai: Mock) -> None:
    mock_mistral_instance = mock_mistralai.return_value
    embeddings_batch_response_mock = MagicMock()
    embeddings_batch_response_mock.data = [MagicMock(embedding=[1.0, 2.0])]
    mock_mistral_instance.embeddings.create.return_value = (
        embeddings_batch_response_mock
    )
    embedder = MistralAIEmbeddings()

    res = embedder.embed_query("my text")

    assert isinstance(res, list)
    assert res == [1.0, 2.0]


@patch("neo4j_graphrag.embeddings.mistral.Mistral")
def test_mistralai_embedder_api_key_via_kwargs(mock_mistral: Mock) -> None:
    mock_mistral_instance = mock_mistral.return_value
    embeddings_batch_response_mock = MagicMock()
    embeddings_batch_response_mock.data = [MagicMock(embedding=[1.0, 2.0])]
    mock_mistral_instance.embeddings.create.return_value = (
        embeddings_batch_response_mock
    )
    api_key = "test_api_key"

    MistralAIEmbeddings(api_key=api_key)

    mock_mistral.assert_called_with(api_key=api_key)


@patch("neo4j_graphrag.embeddings.mistral.Mistral")
@patch("os.getenv")
def test_mistralai_embedder_api_key_from_env(
    mock_getenv: Mock, mock_mistral: Mock
) -> None:
    mock_getenv.return_value = "env_api_key"
    mock_mistral_instance = mock_mistral.return_value
    embeddings_batch_response_mock = MagicMock()
    embeddings_batch_response_mock.data = [MagicMock(embedding=[1.0, 2.0])]
    mock_mistral_instance.embeddings.create.return_value = (
        embeddings_batch_response_mock
    )

    MistralAIEmbeddings()

    mock_getenv.assert_called_with("MISTRAL_API_KEY", "")
    mock_mistral.assert_called_with(api_key="env_api_key")


@patch("neo4j_graphrag.embeddings.mistral.Mistral")
def test_mistralai_embedder_non_retryable_error_handling(mock_mistral: Mock) -> None:
    """Test that non-retryable errors fail immediately without retries."""
    mock_mistral_instance = mock_mistral.return_value
    mock_embeddings = mock_mistral_instance.embeddings.create
    mock_embeddings.side_effect = Exception("API Error")

    embedder = MistralAIEmbeddings()
    # MistralAI now wraps exceptions, so we expect EmbeddingsGenerationError
    with pytest.raises(
        EmbeddingsGenerationError, match="Failed to generate embedding with MistralAI"
    ):
        embedder.embed_query("my text")

    # Verify the API was called only once (no retries for non-rate-limit errors)
    assert mock_embeddings.call_count == 1


@patch("neo4j_graphrag.embeddings.mistral.Mistral")
def test_mistralai_embedder_rate_limit_error_retries(mock_mistral: Mock) -> None:
    """Test that rate limit errors are retried the expected number of times."""
    mock_mistral_instance = mock_mistral.return_value

    # Rate limit error that should trigger retries (matches "too many requests" pattern)
    # Create separate exception instances for each retry attempt
    mock_embeddings = mock_mistral_instance.embeddings.create
    mock_embeddings.side_effect = [
        Exception("too many requests - rate limit exceeded"),
        Exception("too many requests - rate limit exceeded"),
        Exception("too many requests - rate limit exceeded"),
    ]

    embedder = MistralAIEmbeddings()

    # After exhausting retries, tenacity raises RetryError
    with pytest.raises(RetryError):
        embedder.embed_query("my text")

    # Verify the API was called 3 times (default max_attempts for RetryRateLimitHandler)
    assert mock_embeddings.call_count == 3


@patch("neo4j_graphrag.embeddings.mistral.Mistral")
def test_mistralai_embedder_rate_limit_error_eventual_success(
    mock_mistral: Mock,
) -> None:
    """Test that rate limit errors eventually succeed after retries."""
    mock_mistral_instance = mock_mistral.return_value

    # First two calls fail with rate limit, third succeeds
    embeddings_batch_response_mock = MagicMock()
    embeddings_batch_response_mock.data = [MagicMock(embedding=[1.0, 2.0])]

    mock_embeddings = mock_mistral_instance.embeddings.create
    mock_embeddings.side_effect = [
        Exception("too many requests - rate limit exceeded"),
        Exception("too many requests - rate limit exceeded"),
        embeddings_batch_response_mock,
    ]

    embedder = MistralAIEmbeddings()
    result = embedder.embed_query("my text")

    # Verify successful result
    assert result == [1.0, 2.0]
    # Verify the API was called 3 times before succeeding
    assert mock_embeddings.call_count == 3
