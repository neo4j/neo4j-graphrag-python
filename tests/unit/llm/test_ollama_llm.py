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

import ollama
import pytest
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.ollama_llm import OllamaLLM


def get_mock_ollama() -> MagicMock:
    mock = MagicMock()
    mock.ResponseError = ollama.ResponseError
    return mock


@patch("builtins.__import__", side_effect=ImportError)
def test_ollama_llm_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        OllamaLLM(model_name="gpt-4o")


@patch("builtins.__import__")
def test_ollama_llm_happy_path(mock_import: Mock) -> None:
    mock_ollama = get_mock_ollama()
    mock_import.return_value = mock_ollama
    mock_ollama.Client.return_value.chat.return_value = MagicMock(
        message=MagicMock(content="ollama chat response"),
    )
    llm = OllamaLLM(model_name="gpt")

    res = llm.invoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "ollama chat response"
