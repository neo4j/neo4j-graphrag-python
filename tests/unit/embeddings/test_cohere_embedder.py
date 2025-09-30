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
from neo4j_graphrag.embeddings.cohere import CohereEmbeddings
from neo4j_graphrag.exceptions import EmbeddingsGenerationError


@patch("neo4j_graphrag.embeddings.cohere.cohere", None)
def test_cohere_embedder_missing_cohere_dependency() -> None:
    with pytest.raises(ImportError):
        CohereEmbeddings()


@patch("neo4j_graphrag.embeddings.cohere.cohere")
def test_cohere_embedder_happy_path(mock_cohere: Mock) -> None:
    mock_cohere.Client.return_value.embed.return_value = MagicMock(
        embeddings=[[1.0, 2.0]]
    )
    embedder = CohereEmbeddings()
    res = embedder.embed_query("my text")
    assert res == [1.0, 2.0]


@patch("neo4j_graphrag.embeddings.cohere.cohere")
def test_cohere_embedder_non_retryable_error_handling(mock_cohere: Mock) -> None:
    """Test that non-retryable errors fail immediately without retries."""
    mock_embeddings = mock_cohere.Client.return_value.embed
    mock_embeddings.side_effect = Exception("API Error")
    embedder = CohereEmbeddings()
    with pytest.raises(
        EmbeddingsGenerationError, match="Failed to generate embedding with Cohere"
    ):
        embedder.embed_query("my text")

    # Verify the API was called only once (no retries for non-rate-limit errors)
    assert mock_embeddings.call_count == 1


@patch("neo4j_graphrag.embeddings.cohere.cohere")
def test_cohere_embedder_rate_limit_error_retries(mock_cohere: Mock) -> None:
    """Test that rate limit errors are retried the expected number of times."""
    # Rate limit error that should trigger retries (matches "too many requests" pattern)
    # Create separate exception instances for each retry attempt
    mock_embeddings = mock_cohere.Client.return_value.embed
    mock_embeddings.side_effect = [
        Exception("too many requests - please try again later"),
        Exception("too many requests - please try again later"),
        Exception("too many requests - please try again later"),
    ]
    embedder = CohereEmbeddings()

    # After exhausting retries, tenacity raises RetryError
    with pytest.raises(RetryError):
        embedder.embed_query("my text")

    # Verify the API was called 3 times (default max_attempts for RetryRateLimitHandler)
    assert mock_cohere.Client.return_value.embed.call_count == 3


@patch("neo4j_graphrag.embeddings.cohere.cohere")
def test_cohere_embedder_rate_limit_error_eventual_success(mock_cohere: Mock) -> None:
    """Test that rate limit errors eventually succeed after retries."""
    # First two calls fail with rate limit, third succeeds
    mock_embeddings = mock_cohere.Client.return_value.embed
    mock_embeddings.side_effect = [
        Exception("too many requests - please try again later"),
        Exception("too many requests - please try again later"),
        MagicMock(embeddings=[[1.0, 2.0]]),
    ]
    embedder = CohereEmbeddings()

    result = embedder.embed_query("my text")

    # Verify successful result
    assert result == [1.0, 2.0]
    # Verify the API was called 3 times before succeeding
    assert mock_embeddings.call_count == 3
