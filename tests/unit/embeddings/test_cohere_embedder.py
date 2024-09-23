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
from neo4j_graphrag.embeddings.cohere import CohereEmbeddings


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
