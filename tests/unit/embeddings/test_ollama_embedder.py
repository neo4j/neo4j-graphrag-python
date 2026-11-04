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
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.exceptions import EmbeddingsGenerationError


@patch("builtins.__import__", side_effect=ImportError)
def test_ollama_embedder_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        OllamaEmbeddings(model="test")


@patch("builtins.__import__")
def test_ollama_embedder_happy_path(mock_import: Mock) -> None:
    mock_import.return_value.Client.return_value.embed.return_value = MagicMock(
        embeddings=[[1.0, 2.0]],
    )
    embedder = OllamaEmbeddings(model="test")
    res = embedder.embed_query("my text")
    assert isinstance(res, list)
    assert res == [1.0, 2.0]


@patch("builtins.__import__")
def test_ollama_embedder_empty_list(mock_import: Mock) -> None:
    mock_import.return_value.Client.return_value.embed.return_value = MagicMock(
        embeddings=[],
    )
    embedder = OllamaEmbeddings(model="test")
    with pytest.raises(EmbeddingsGenerationError):
        embedder.embed_query("my text")


@patch("builtins.__import__")
def test_ollama_embedding_model_name_is_string(mock_import: Mock) -> None:
    embedder = OllamaEmbeddings(model="test")
    assert isinstance(embedder.embedding_model_name, str)
    assert embedder.embedding_model_name == "test"


@patch("builtins.__import__")
def test_ollama_embed_documents_single_call(mock_import: Mock) -> None:
    embed = mock_import.return_value.Client.return_value.embed
    embed.return_value = MagicMock(embeddings=[[1.0], [2.0]])
    embedder = OllamaEmbeddings(model="test")
    assert embedder.embed_documents(["a", "b"]) == [[1.0], [2.0]]
    embed.assert_called_once_with(model="test", input=["a", "b"])


@patch("builtins.__import__")
def test_ollama_embed_documents_empty_and_errors(mock_import: Mock) -> None:
    embed = mock_import.return_value.Client.return_value.embed
    embedder = OllamaEmbeddings(model="test")
    assert embedder.embed_documents([]) == []
    embed.assert_not_called()

    embed.return_value = MagicMock(embeddings=[[1.0]])
    with pytest.raises(EmbeddingsGenerationError, match="1 embeddings for 2 inputs"):
        embedder.embed_documents(["a", "b"])

    embed.return_value = MagicMock(embeddings=[])
    with pytest.raises(EmbeddingsGenerationError, match="Failed to retrieve"):
        embedder.embed_documents(["a"])

    embed.return_value = MagicMock(embeddings=["oops"])
    with pytest.raises(EmbeddingsGenerationError, match="not a list of floats"):
        embedder.embed_documents(["a"])


@pytest.mark.asyncio
@patch("builtins.__import__")
async def test_ollama_async_embed_documents(mock_import: Mock) -> None:
    async_embed = AsyncMock(return_value=MagicMock(embeddings=[[1.0], [2.0]]))
    mock_import.return_value.AsyncClient.return_value.embed = async_embed
    embedder = OllamaEmbeddings(model="test")
    assert await embedder.async_embed_documents(["a", "b"]) == [[1.0], [2.0]]
    assert await embedder.async_embed_documents([]) == []
    async_embed.assert_awaited_once_with(model="test", input=["a", "b"])
