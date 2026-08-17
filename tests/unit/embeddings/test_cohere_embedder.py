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


@patch("neo4j_graphrag.embeddings.cohere.cohere")
def test_cohere_embedder_sends_input_type(mock_cohere: Mock) -> None:
    """Current Cohere models reject a request that omits input_type."""
    mock_cohere.Client.return_value.embed.return_value = MagicMock(
        embeddings=[[1.0, 2.0]]
    )
    embedder = CohereEmbeddings(model="embed-english-v3.0")
    embedder.embed_query("my text")

    _, kwargs = mock_cohere.Client.return_value.embed.call_args
    assert kwargs["input_type"] == "search_document"


@patch("neo4j_graphrag.embeddings.cohere.cohere")
def test_cohere_embedder_input_type_from_constructor(mock_cohere: Mock) -> None:
    mock_cohere.Client.return_value.embed.return_value = MagicMock(
        embeddings=[[1.0, 2.0]]
    )
    embedder = CohereEmbeddings(model="embed-english-v3.0", input_type="search_query")
    embedder.embed_query("my text")

    _, kwargs = mock_cohere.Client.return_value.embed.call_args
    assert kwargs["input_type"] == "search_query"


@patch("neo4j_graphrag.embeddings.cohere.cohere")
def test_cohere_embedder_second_positional_arg_is_still_the_rate_limit_handler(
    mock_cohere: Mock,
) -> None:
    """input_type must not displace rate_limit_handler.

    Adding it as the second positional parameter would silently rebind an
    existing ``CohereEmbeddings(model, handler)`` call: the handler would be
    dropped and passed to the API as input_type. It is keyword-only for that
    reason, matching (model, rate_limit_handler, **kwargs) on the other embedders.
    """
    handler = MagicMock()
    embedder = CohereEmbeddings("embed-english-v3.0", handler)

    assert embedder._rate_limit_handler is handler
    assert embedder.input_type == "search_document"


@patch("neo4j_graphrag.embeddings.cohere.cohere")
def test_cohere_embedder_input_type_cannot_be_passed_positionally(
    mock_cohere: Mock,
) -> None:
    with pytest.raises(TypeError):
        CohereEmbeddings("embed-english-v3.0", None, "search_query")  # type: ignore[misc]


@patch("neo4j_graphrag.embeddings.cohere.cohere")
def test_cohere_embedder_per_call_input_type_wins(mock_cohere: Mock) -> None:
    """A per-call value overrides the constructor's without colliding with it."""
    mock_cohere.Client.return_value.embed.return_value = MagicMock(
        embeddings=[[1.0, 2.0]]
    )
    embedder = CohereEmbeddings(model="embed-english-v3.0")
    embedder.embed_query("my text", input_type="classification")

    _, kwargs = mock_cohere.Client.return_value.embed.call_args
    assert kwargs["input_type"] == "classification"
