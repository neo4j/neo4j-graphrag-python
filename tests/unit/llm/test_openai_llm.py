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
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.openai import AzureOpenAILLM, OpenAILLM


@patch("neo4j_graphrag.llm.openai.openai", None)
def test_openai_llm_missing_dependency() -> None:
    with pytest.raises(ImportError):
        OpenAILLM(model_name="gpt-4o")


@patch("neo4j_graphrag.llm.openai.OpenAILLM.client_class")
def test_openai_llm_happy_path(mock_openai: Mock) -> None:
    mock_openai.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="openai chat response"))],
    )
    llm = OpenAILLM(api_key="my key", model_name="gpt")
    res = llm.invoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "openai chat response"


@patch("neo4j_graphrag.llm.openai.openai", None)
def test_azure_openai_llm_missing_dependency() -> None:
    with pytest.raises(ImportError):
        AzureOpenAILLM(model_name="gpt-4o")


@patch("neo4j_graphrag.llm.openai.AzureOpenAILLM.client_class")
def test_azure_openai_llm_happy_path(mock_openai: Mock) -> None:
    mock_openai.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="openai chat response"))],
    )
    llm = AzureOpenAILLM(
        model_name="gpt",
        azure_endpoint="https://test.openai.azure.com/",
        api_key="my key",
        api_version="version",
    )
    res = llm.invoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "openai chat response"
