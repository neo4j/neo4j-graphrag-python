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
from neo4j_graphrag.embeddings.vertexai import VertexAIEmbeddings


@patch("neo4j_graphrag.embeddings.vertexai.vertexai", None)
def test_vertexai_embedder_missing_dependency() -> None:
    with pytest.raises(ImportError):
        VertexAIEmbeddings()


@patch("neo4j_graphrag.embeddings.vertexai.vertexai")
def test_vertexai_embedder_happy_path(mock_vertexai: Mock) -> None:
    mock_vertexai.language_models.TextEmbeddingModel.from_pretrained.return_value.get_embeddings.return_value = [
        MagicMock(values=[1.0, 2.0])
    ]
    embedder = VertexAIEmbeddings()
    res = embedder.embed_query("my text")
    assert isinstance(res, list)
    assert res == [1.0, 2.0]
