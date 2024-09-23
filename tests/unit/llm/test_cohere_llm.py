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
from unittest.mock import Mock, patch

import pytest
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.cohere import CohereLLM


@patch("neo4j_graphrag.embeddings.cohere.cohere", None)
def test_cohere_llm_missing_dependency() -> None:
    with pytest.raises(ImportError):
        CohereLLM(model_name="something")


@patch("neo4j_graphrag.llm.cohere.cohere")
def test_cohere_embedder_happy_path(mock_cohere: Mock) -> None:
    mock_cohere.Cohere.return_value.chat.return_value = []
    embedder = CohereLLM(model_name="something")
    res = embedder.invoke("my text")
    assert isinstance(res, LLMResponse)
    assert False
