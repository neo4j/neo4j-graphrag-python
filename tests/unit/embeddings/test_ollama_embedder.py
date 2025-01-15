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
