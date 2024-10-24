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
from neo4j_graphrag.embeddings.openai import (
    AzureOpenAIEmbeddings,
    OpenAIEmbeddings,
)
import openai


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
    with patch(
        "neo4j_graphrag.embeddings.openai.openai.OpenAI"
    ) as mock_openai_client, patch(
        "neo4j_graphrag.embeddings.openai.openai.AzureOpenAI"
    ) as mock_azure_openai_client:
        mock_azure_openai_client.return_value = MagicMock()

        AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            azure_endpoint="https://test.openai.azure.com/",
            api_key="my_key",
            api_version="2023-05-15",
        )

        mock_openai_client.assert_not_called()
        mock_azure_openai_client.assert_called_once_with(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="my_key",
            api_version="2023-05-15",
        )
