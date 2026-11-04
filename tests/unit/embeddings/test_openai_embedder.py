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

import openai
import pytest
from tenacity import RetryError
from neo4j_graphrag.embeddings.openai import (
    AzureOpenAIEmbeddings,
    OpenAIEmbeddings,
)
from neo4j_graphrag.exceptions import EmbeddingsGenerationError


def get_mock_openai() -> MagicMock:
    mock = MagicMock()
    mock.OpenAIError = openai.OpenAIError
    return mock


@patch("builtins.__import__", side_effect=ImportError)
def test_openai_embedder_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        OpenAIEmbeddings()


@patch("builtins.__import__")
def test_openai_embedder_happy_path(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai

    mock_openai.OpenAI.return_value.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[1.0, 2.0])],
    )
    embedder = OpenAIEmbeddings(api_key="my key")
    res = embedder.embed_query("my text")
    assert isinstance(res, list)
    assert res == [1.0, 2.0]


@patch("builtins.__import__", side_effect=ImportError)
def test_azure_openai_embedder_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        AzureOpenAIEmbeddings()


@patch("builtins.__import__")
def test_azure_openai_embedder_happy_path(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai

    mock_openai.AzureOpenAI.return_value.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[1.0, 2.0])],
    )
    embedder = AzureOpenAIEmbeddings(
        model_name="gpt",
        azure_endpoint="https://test.openai.azure.com/",
        api_key="my key",
        api_version="version",
    )
    res = embedder.embed_query("my text")
    assert isinstance(res, list)
    assert res == [1.0, 2.0]


def test_azure_openai_embedder_does_not_call_openai_client() -> None:
    from unittest.mock import patch

    mock_openai = get_mock_openai()

    with patch.dict("sys.modules", {"openai": mock_openai}):
        AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            azure_endpoint="https://test.openai.azure.com/",
            api_key="my_key",
            api_version="2023-05-15",
        )

        mock_openai.OpenAI.assert_not_called()
        mock_openai.AzureOpenAI.assert_called_once_with(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="my_key",
            api_version="2023-05-15",
        )


@patch("builtins.__import__")
def test_openai_embedder_non_retryable_error_handling(mock_import: Mock) -> None:
    """Test that non-retryable errors fail immediately without retries."""
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai

    # Generic API error that doesn't match rate limit patterns - should not be retried
    mock_embeddings = mock_openai.OpenAI.return_value.embeddings.create
    mock_embeddings.side_effect = Exception("API Error")
    embedder = OpenAIEmbeddings(api_key="my key")

    with pytest.raises(
        EmbeddingsGenerationError, match="Failed to generate embedding with OpenAI"
    ):
        embedder.embed_query("my text")

    # Verify the API was called only once (no retries for non-rate-limit errors)
    assert mock_embeddings.call_count == 1


@patch("builtins.__import__")
def test_openai_embedder_rate_limit_error_retries(mock_import: Mock) -> None:
    """Test that rate limit errors are retried the expected number of times."""
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai

    # Rate limit error that should trigger retries (matches "429" pattern)
    # Create separate exception instances for each retry attempt
    mock_embeddings = mock_openai.OpenAI.return_value.embeddings.create
    mock_embeddings.side_effect = [
        Exception("Error code: 429 - Too many requests"),
        Exception("Error code: 429 - Too many requests"),
        Exception("Error code: 429 - Too many requests"),
    ]
    embedder = OpenAIEmbeddings(api_key="my key")

    # After exhausting retries, tenacity raises RetryError
    with pytest.raises(RetryError):
        embedder.embed_query("my text")

    # Verify the API was called 3 times (default max_attempts for RetryRateLimitHandler)
    assert mock_embeddings.call_count == 3


@patch("builtins.__import__")
def test_openai_embedder_rate_limit_error_eventual_success(mock_import: Mock) -> None:
    """Test that rate limit errors eventually succeed after retries."""
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai

    # First two calls fail with rate limit, third succeeds
    mock_embeddings = mock_openai.OpenAI.return_value.embeddings.create
    mock_embeddings.side_effect = [
        Exception("Error code: 429 - Too many requests"),
        Exception("Error code: 429 - Too many requests"),
        MagicMock(data=[MagicMock(embedding=[1.0, 2.0])]),
    ]
    embedder = OpenAIEmbeddings(api_key="my key")

    result = embedder.embed_query("my text")

    # Verify successful result
    assert result == [1.0, 2.0]
    # Verify the API was called 3 times before succeeding
    assert mock_embeddings.call_count == 3


@patch("builtins.__import__")
def test_openai_embedding_model_name_is_string(mock_import: Mock) -> None:
    embedder = OpenAIEmbeddings(api_key="my key")
    assert isinstance(embedder.embedding_model_name, str)
    assert embedder.embedding_model_name == "text-embedding-ada-002"


def _openai_response(vectors: list[list[float]], indexes: list[int]) -> MagicMock:
    return MagicMock(
        data=[MagicMock(embedding=v, index=i) for v, i in zip(vectors, indexes)]
    )


@patch("builtins.__import__")
def test_openai_embed_documents_single_request(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    create = mock_openai.OpenAI.return_value.embeddings.create
    # out-of-order response: results must be mapped back by index
    create.return_value = _openai_response([[2.0], [1.0], [3.0]], [1, 0, 2])
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    res = embedder.embed_documents(["a", "b", "c"], dimensions=1)
    assert res == [[1.0], [2.0], [3.0]]
    create.assert_called_once_with(
        input=["a", "b", "c"], model="text-embedding-3-small", dimensions=1
    )


@patch("builtins.__import__")
def test_openai_embed_documents_splits_at_request_limit(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    create = mock_openai.OpenAI.return_value.embeddings.create

    def fake_create(input: list[str], model: str) -> MagicMock:
        return _openai_response([[float(t)] for t in input], list(range(len(input))))

    create.side_effect = fake_create
    embedder = OpenAIEmbeddings()
    embedder.max_inputs_per_request = 2
    res = embedder.embed_documents(["1", "2", "3", "4", "5"])
    assert res == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert [c.kwargs["input"] for c in create.call_args_list] == [
        ["1", "2"],
        ["3", "4"],
        ["5"],
    ]


@patch("builtins.__import__")
def test_openai_embed_documents_empty(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    embedder = OpenAIEmbeddings()
    assert embedder.embed_documents([]) == []
    mock_openai.OpenAI.return_value.embeddings.create.assert_not_called()


@patch("builtins.__import__")
def test_openai_embed_documents_count_mismatch(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.OpenAI.return_value.embeddings.create.return_value = _openai_response(
        [[1.0]], [0]
    )
    embedder = OpenAIEmbeddings()
    with pytest.raises(EmbeddingsGenerationError, match="1 embeddings for 2 inputs"):
        embedder.embed_documents(["a", "b"])


@patch("builtins.__import__")
def test_openai_embed_documents_rate_limit_retries_only_failed_group(
    mock_import: Mock,
) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    create = mock_openai.OpenAI.return_value.embeddings.create
    create.side_effect = [
        _openai_response([[1.0]], [0]),
        Exception("Error code: 429 - Too many requests"),
        _openai_response([[2.0]], [0]),
    ]
    embedder = OpenAIEmbeddings()
    embedder.max_inputs_per_request = 1
    with patch("time.sleep"):
        res = embedder.embed_documents(["a", "b"])
    assert res == [[1.0], [2.0]]
    assert [c.kwargs["input"] for c in create.call_args_list] == [["a"], ["b"], ["b"]]


@patch("builtins.__import__")
def test_openai_embed_documents_api_error(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.OpenAI.return_value.embeddings.create.side_effect = Exception(
        "API Error"
    )
    embedder = OpenAIEmbeddings()
    with pytest.raises(
        EmbeddingsGenerationError, match="Failed to generate embeddings with OpenAI"
    ):
        embedder.embed_documents(["a"])


@pytest.mark.asyncio
@patch("builtins.__import__")
async def test_openai_async_embed_documents(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.OpenAI.return_value.embeddings.create.return_value = _openai_response(
        [[1.0], [2.0]], [0, 1]
    )
    embedder = OpenAIEmbeddings()
    assert await embedder.async_embed_documents(["a", "b"]) == [[1.0], [2.0]]


@patch("builtins.__import__")
def test_azure_openai_embed_documents(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    create = mock_openai.AzureOpenAI.return_value.embeddings.create
    create.return_value = _openai_response([[1.0], [2.0]], [0, 1])
    embedder = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_endpoint="https://test.openai.azure.com/",
        api_key="my_key",
        api_version="2024-02-01",
    )
    assert embedder.embed_documents(["a", "b"]) == [[1.0], [2.0]]
    create.assert_called_once_with(input=["a", "b"], model="text-embedding-3-small")
