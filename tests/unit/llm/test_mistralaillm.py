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
from neo4j_graphrag.embeddings.mistral import MistralAIEmbeddings


@patch("neo4j_graphrag.embeddings.mistral.Mistral", None)
def test_mistralai_embeddings_missing_dependency() -> None:
    with pytest.raises(ImportError):
        MistralAIEmbeddings()


@patch("neo4j_graphrag.embeddings.mistral.Mistral")
def test_mistralai_embeddings_happy_path(mock_mistral: Mock) -> None:
    mock_mistral_instance = mock_mistral.return_value
    embeddings_batch_response_mock = MagicMock()
    embeddings_batch_response_mock.data = [MagicMock(embedding=[1.0, 2.0, 3.0])]
    mock_mistral_instance.embeddings.create.return_value = embeddings_batch_response_mock
    embedder = MistralAIEmbeddings()

    res = embedder.embed_query("some text")

    assert isinstance(res, list)
    assert res == [1.0, 2.0, 3.0]
