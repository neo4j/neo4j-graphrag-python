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
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from mistralai.models.sdkerror import SDKError
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMResponse, MistralAILLM


pytestmark = pytest.mark.mistralai


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral", None)
def test_mistral_ai_llm_missing_dependency() -> None:
    with pytest.raises(ImportError):
        MistralAILLM(model_name="mistral-model")


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistral_ai_llm_invoke(mock_mistral: Mock) -> None:
    mock_mistral_instance = mock_mistral.return_value

    chat_response_mock = MagicMock()
    chat_response_mock.choices = [
        MagicMock(message=MagicMock(content="mistral response"))
    ]

    mock_mistral_instance.chat.complete.return_value = chat_response_mock

    llm = MistralAILLM(model_name="mistral-model")

    res = llm.invoke("some input")

    assert isinstance(res, LLMResponse)
    assert res.content == "mistral response"


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
async def test_mistral_ai_llm_ainvoke(mock_mistral: Mock) -> None:
    mock_mistral_instance = mock_mistral.return_value

    async def mock_complete_async(*args: Any, **kwargs: Any) -> MagicMock:
        chat_response_mock = MagicMock()
        chat_response_mock.choices = [
            MagicMock(message=MagicMock(content="async mistral response"))
        ]
        return chat_response_mock

    mock_mistral_instance.chat.complete_async = mock_complete_async

    llm = MistralAILLM(model_name="mistral-model")

    res = await llm.ainvoke("some input")

    assert isinstance(res, LLMResponse)
    assert res.content == "async mistral response"


@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
def test_mistral_ai_llm_invoke_sdkerror(mock_mistral: Mock) -> None:
    mock_mistral_instance = mock_mistral.return_value
    mock_mistral_instance.chat.complete.side_effect = SDKError("Some error")

    llm = MistralAILLM(model_name="mistral-model")

    with pytest.raises(LLMGenerationError):
        llm.invoke("some input")


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.mistralai_llm.Mistral")
async def test_mistral_ai_llm_ainvoke_sdkerror(mock_mistral: Mock) -> None:
    mock_mistral_instance = mock_mistral.return_value

    async def mock_complete_async(*args: Any, **kwargs: Any) -> None:
        raise SDKError("Some async error")

    mock_mistral_instance.chat.complete_async = mock_complete_async

    llm = MistralAILLM(model_name="mistral-model")

    with pytest.raises(LLMGenerationError):
        await llm.ainvoke("some input")
