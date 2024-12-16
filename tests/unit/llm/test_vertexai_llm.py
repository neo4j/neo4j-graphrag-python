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
from __future__ import annotations

from unittest import mock
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.vertexai_llm import VertexAILLM
from vertexai.generative_models import Content, Part


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel", None)
def test_vertexai_llm_missing_dependency() -> None:
    with pytest.raises(ImportError):
        VertexAILLM(model_name="gemini-1.5-flash-001")


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_invoke_happy_path(GenerativeModelMock: MagicMock) -> None:
    mock_response = Mock()
    mock_response.text = "Return text"
    mock_model = GenerativeModelMock.return_value
    mock_model.generate_content.return_value = mock_response
    model_params = {"temperature": 0.5}
    llm = VertexAILLM("gemini-1.5-flash-001", model_params)
    input_text = "may thy knife chip and shatter"
    response = llm.invoke(input_text)
    assert response.content == "Return text"
    llm.model.generate_content.assert_called_once_with([mock.ANY], **model_params)


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_get_messages(GenerativeModelMock: MagicMock) -> None:
    system_instruction = "You are a helpful assistant."
    model_name = "gemini-1.5-flash-001"
    question = "When does it set?"
    message_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
        {"role": "user", "content": "What about next season?"},
        {"role": "assistant", "content": "Around 8am."},
    ]
    expected_response = [
        Content(
            role="user",
            parts=[Part.from_text("When does the sun come up in the summer?")],
        ),
        Content(role="model", parts=[Part.from_text("Usually around 6am.")]),
        Content(role="user", parts=[Part.from_text("What about next season?")]),
        Content(role="model", parts=[Part.from_text("Around 8am.")]),
        Content(role="user", parts=[Part.from_text("When does it set?")]),
    ]

    llm = VertexAILLM(model_name=model_name, system_instruction=system_instruction)
    response = llm.get_messages(question, message_history)

    GenerativeModelMock.assert_called_once_with(
        model_name=model_name, system_instruction=[system_instruction]
    )
    assert len(response) == len(expected_response)
    for actual, expected in zip(response, expected_response):
        assert actual.role == expected.role
        assert actual.parts[0].text == expected.parts[0].text


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_get_messages_validation_error(GenerativeModelMock: MagicMock) -> None:
    system_instruction = "You are a helpful assistant."
    model_name = "gemini-1.5-flash-001"
    question = "hi!"
    message_history = [
        {"role": "model", "content": "hello!"},
    ]

    llm = VertexAILLM(model_name=model_name, system_instruction=system_instruction)
    with pytest.raises(LLMGenerationError) as exc_info:
        llm.invoke(question, message_history)
    assert "Input should be 'user' or 'assistant'" in str(exc_info.value)


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
async def test_vertexai_ainvoke_happy_path(GenerativeModelMock: MagicMock) -> None:
    mock_response = AsyncMock()
    mock_response.text = "Return text"
    mock_model = GenerativeModelMock.return_value
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    model_params = {"temperature": 0.5}
    llm = VertexAILLM("gemini-1.5-flash-001", model_params)
    input_text = "may thy knife chip and shatter"
    response = await llm.ainvoke(input_text)
    assert response.content == "Return text"
    llm.model.generate_content_async.assert_called_once_with([mock.ANY], **model_params)
