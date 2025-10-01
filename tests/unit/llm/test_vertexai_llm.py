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

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.types import ToolCallResponse
from neo4j_graphrag.llm.vertexai_llm import VertexAILLM
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage
from vertexai.generative_models import (
    Content,
    GenerationResponse,
    Part,
)


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel", None)
def test_vertexai_llm_missing_dependency() -> None:
    with pytest.raises(ImportError):
        VertexAILLM(model_name="gemini-1.5-flash-001")


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_invoke_happy_path(GenerativeModelMock: MagicMock) -> None:
    model_name = "gemini-1.5-flash-001"
    input_text = "may thy knife chip and shatter"
    mock_response = Mock()
    mock_response.text = "Return text"
    mock_model = GenerativeModelMock.return_value
    mock_model.generate_content.return_value = mock_response
    model_params = {"temperature": 0.5}
    llm = VertexAILLM(model_name, model_params)

    response = llm.invoke(input_text)
    assert response.content == "Return text"
    GenerativeModelMock.assert_called_once_with(
        model_name=model_name,
        system_instruction=None,
    )
    last_call = mock_model.generate_content.call_args_list[0]
    content = last_call.kwargs["contents"]
    assert len(content) == 1
    assert content[0].role == "user"
    assert content[0].parts[0].text == input_text


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_get_messages(GenerativeModelMock: MagicMock) -> None:
    model_name = "gemini-1.5-flash-001"
    message_history: list[LLMMessage] = [
        {"role": "system", "content": "Answer to a 3yo kid"},
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
        {"role": "user", "content": "What about next season?"},
    ]
    expected_response = [
        Content(
            role="user",
            parts=[Part.from_text("When does the sun come up in the summer?")],
        ),
        Content(role="model", parts=[Part.from_text("Usually around 6am.")]),
        Content(role="user", parts=[Part.from_text("What about next season?")]),
    ]

    llm = VertexAILLM(model_name=model_name)
    system_instructions, messages = llm.get_messages(message_history)

    GenerativeModelMock.assert_not_called()
    assert system_instructions == "Answer to a 3yo kid"
    assert len(messages) == len(expected_response)
    for actual, expected in zip(messages, expected_response):
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
    with pytest.raises(LLMGenerationError, match="Input validation failed"):
        llm.invoke(question, message_history)


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
@patch("neo4j_graphrag.llm.vertexai_llm.VertexAILLM.get_messages")
async def test_vertexai_ainvoke_happy_path(
    mock_get_messages: Mock, GenerativeModelMock: MagicMock
) -> None:
    mock_response = AsyncMock()
    mock_response.text = "Return text"
    mock_model = GenerativeModelMock.return_value
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    mock_get_messages.return_value = None, [{"text": "Return text"}]
    model_params = {"temperature": 0.5}
    llm = VertexAILLM("gemini-1.5-flash-001", model_params)
    input_text = "may thy knife chip and shatter"
    response = await llm.ainvoke(input_text)
    assert response.content == "Return text"
    mock_model.generate_content_async.assert_awaited_once_with(
        contents=[{"text": "Return text"}]
    )


def test_vertexai_get_llm_tools(test_tool: Tool) -> None:
    llm = VertexAILLM(model_name="gemini")
    tools = llm._get_llm_tools(tools=[test_tool])
    assert tools is not None
    assert len(tools) == 1
    tool = tools[0]
    tool_dict = tool.to_dict()
    assert len(tool_dict["function_declarations"]) == 1
    assert tool_dict["function_declarations"][0]["name"] == "test_tool"
    assert tool_dict["function_declarations"][0]["description"] == "A test tool"


@patch("neo4j_graphrag.llm.vertexai_llm.VertexAILLM._parse_tool_response")
@patch("neo4j_graphrag.llm.vertexai_llm.VertexAILLM._call_llm")
def test_vertexai_invoke_with_tools(
    mock_call_llm: Mock,
    mock_parse_tool: Mock,
    test_tool: Tool,
) -> None:
    # Mock the model call response
    tool_call_mock = MagicMock()
    tool_call_mock.name = "function"
    tool_call_mock.args = {}
    mock_call_llm.return_value = MagicMock(
        candidates=[MagicMock(function_calls=[tool_call_mock])]
    )
    mock_parse_tool.return_value = ToolCallResponse(tool_calls=[])

    llm = VertexAILLM(model_name="gemini")
    tools = [test_tool]

    res = llm.invoke_with_tools("my text", tools)
    mock_call_llm.assert_called_once_with(
        [{"role": "user", "content": "my text"}],
        tools=tools,
    )
    mock_parse_tool.assert_called_once()
    assert isinstance(res, ToolCallResponse)


@patch("neo4j_graphrag.llm.vertexai_llm.VertexAILLM._get_model")
def test_vertexai_call_llm_with_tools(mock_model: Mock, test_tool: Tool) -> None:
    # Mock the generation response
    mock_generate_content = mock_model.return_value.generate_content
    mock_generate_content.return_value = MagicMock(
        spec=GenerationResponse,
    )

    llm = VertexAILLM(model_name="gemini")
    tools = [test_tool]

    with patch.object(llm, "_get_llm_tools", return_value=["my tools"]):
        res = llm._call_llm([{"role": "user", "content": "my text"}], tools=tools)
        assert isinstance(res, GenerationResponse)

        mock_model.assert_called_once_with(
            None,
        )
        calls = mock_generate_content.call_args_list
        assert len(calls) == 1
        assert calls[0][1]["tools"] == ["my tools"]
        assert calls[0][1]["tool_config"] is not None


@patch("neo4j_graphrag.llm.vertexai_llm.VertexAILLM._parse_tool_response")
@patch("neo4j_graphrag.llm.vertexai_llm.VertexAILLM._call_llm")
def test_vertexai_ainvoke_with_tools(
    mock_call_llm: Mock,
    mock_parse_tool: Mock,
    test_tool: Tool,
) -> None:
    # Mock the model call response
    tool_call_mock = MagicMock()
    tool_call_mock.name = "function"
    tool_call_mock.args = {}
    mock_call_llm.return_value = AsyncMock(
        return_value=MagicMock(candidates=[MagicMock(function_calls=[tool_call_mock])])
    )
    mock_parse_tool.return_value = ToolCallResponse(tool_calls=[])

    llm = VertexAILLM(model_name="gemini")
    tools = [test_tool]

    res = llm.invoke_with_tools("my text", tools)
    mock_call_llm.assert_called_once_with(
        [{"role": "user", "content": "my text"}],
        tools=tools,
    )
    mock_parse_tool.assert_called_once()
    assert isinstance(res, ToolCallResponse)


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.vertexai_llm.VertexAILLM._get_model")
async def test_vertexai_acall_llm_with_tools(mock_model: Mock, test_tool: Tool) -> None:
    # Mock the generation response
    mock_model.return_value = AsyncMock(
        generate_content_async=AsyncMock(
            return_value=MagicMock(
                spec=GenerationResponse,
            )
        )
    )

    llm = VertexAILLM(model_name="gemini")
    tools = [test_tool]

    res = await llm._acall_llm([{"role": "user", "content": "my text"}], tools=tools)
    mock_model.assert_called_once_with(
        None,
    )
    assert isinstance(res, GenerationResponse)
