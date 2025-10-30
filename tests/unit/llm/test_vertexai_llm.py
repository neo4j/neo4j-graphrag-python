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

from typing import cast
from typing import List

from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest
from vertexai.generative_models import (
    Content,
    GenerationResponse,
    Part,
)

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.types import ToolCallResponse
from neo4j_graphrag.llm.vertexai_llm import VertexAILLM
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage


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
@patch("neo4j_graphrag.llm.vertexai_llm.VertexAILLM.get_messages")
def test_vertexai_invoke_with_system_instruction(
    mock_get_messages: MagicMock,
    GenerativeModelMock: MagicMock,
) -> None:
    system_instruction = "You are a helpful assistant."
    model_name = "gemini-1.5-flash-001"
    input_text = "may thy knife chip and shatter"
    mock_response = Mock()
    mock_response.text = "Return text"
    mock_model = GenerativeModelMock.return_value
    mock_model.generate_content.return_value = mock_response

    mock_get_messages.return_value = [{"text": "some text"}]

    model_params = {"temperature": 0.5}
    llm = VertexAILLM(model_name, model_params)

    response = llm.invoke(input_text, system_instruction=system_instruction)
    assert response.content == "Return text"
    GenerativeModelMock.assert_called_once_with(
        model_name=model_name,
        system_instruction=system_instruction,
    )
    mock_model.generate_content.assert_called_once_with(
        contents=[{"text": "some text"}]
    )


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_invoke_with_message_history_and_system_instruction(
    GenerativeModelMock: MagicMock,
) -> None:
    system_instruction = "You are a helpful assistant."
    model_name = "gemini-1.5-flash-001"
    mock_response = Mock()
    mock_response.text = "Return text"
    mock_model = GenerativeModelMock.return_value
    mock_model.generate_content.return_value = mock_response
    model_params = {"temperature": 0.5}
    llm = VertexAILLM(model_name, model_params)

    message_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    response = llm.invoke(
        question,
        message_history,  # type: ignore
        system_instruction=system_instruction,
    )
    assert response.content == "Return text"
    GenerativeModelMock.assert_called_once_with(
        model_name=model_name,
        system_instruction=system_instruction,
    )
    last_call = mock_model.generate_content.call_args_list[0]
    content = last_call.kwargs["contents"]
    assert len(content) == 3  # question + 2 messages in history


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_get_messages(GenerativeModelMock: MagicMock) -> None:
    model_name = "gemini-1.5-flash-001"
    question = "When does it set?"
    message_history: list[LLMMessage] = [
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

    llm = VertexAILLM(model_name=model_name)
    response = llm.get_messages(question, message_history)

    GenerativeModelMock.assert_not_called()
    assert len(response) == len(expected_response)
    for actual, expected in zip(response, expected_response):
        assert actual.role == expected.role
        assert actual.parts[0].text == expected.parts[0].text


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_get_messages_validation_error(
    _GenerativeModelMock: MagicMock,
) -> None:
    system_instruction = "You are a helpful assistant."
    model_name = "gemini-1.5-flash-001"
    question = "hi!"
    message_history = [
        {"role": "model", "content": "hello!"},
    ]

    llm = VertexAILLM(model_name=model_name, system_instruction=system_instruction)
    with pytest.raises(LLMGenerationError) as exc_info:
        llm.invoke(question, cast(list[LLMMessage], message_history))
    assert "Input should be 'user', 'assistant' or 'system" in str(exc_info.value)


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
    mock_get_messages.return_value = [{"text": "Return text"}]
    model_params = {"temperature": 0.5}
    llm = VertexAILLM("gemini-1.5-flash-001", model_params)
    input_text = "may thy knife chip and shatter"
    response = await llm.ainvoke(input_text)
    print(f"Response: {response}")
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
        "my text",
        message_history=None,
        system_instruction=None,
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
        res = llm._call_llm("my text", tools=tools)
        assert isinstance(res, GenerationResponse)

        mock_model.assert_called_once_with(
            system_instruction=None,
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
        "my text",
        message_history=None,
        system_instruction=None,
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

    res = await llm._acall_llm("my text", tools=tools)
    mock_model.assert_called_once_with(
        system_instruction=None,
    )
    assert isinstance(res, GenerationResponse)


# LLM Interface V2 Tests


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_invoke_v2_happy_path(GenerativeModelMock: MagicMock) -> None:
    """Test V2 interface invoke method with List[LLMMessage] input."""
    model_name = "gemini-1.5-flash-001"
    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    mock_response = Mock()
    mock_response.text = "Paris is the capital of France."
    mock_model = GenerativeModelMock.return_value
    mock_model.generate_content.return_value = mock_response

    llm = VertexAILLM(model_name=model_name)
    response = llm.invoke(messages)

    assert response.content == "Paris is the capital of France."
    GenerativeModelMock.assert_called_once_with(
        model_name=model_name,
        system_instruction="You are a helpful assistant.",
    )
    mock_model.generate_content.assert_called_once()
    call_args = mock_model.generate_content.call_args
    contents = call_args.kwargs["contents"]
    assert len(contents) == 1  # Only user message after system is extracted
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "What is the capital of France?"


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_invoke_v2_with_conversation_history(
    GenerativeModelMock: MagicMock,
) -> None:
    """Test V2 interface invoke with conversation history."""
    model_name = "gemini-1.5-flash-001"
    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris is the capital of France."},
        {"role": "user", "content": "What about Germany?"},
    ]
    mock_response = Mock()
    mock_response.text = "Berlin is the capital of Germany."
    mock_model = GenerativeModelMock.return_value
    mock_model.generate_content.return_value = mock_response

    llm = VertexAILLM(model_name=model_name)
    response = llm.invoke(messages)

    assert response.content == "Berlin is the capital of Germany."
    GenerativeModelMock.assert_called_once_with(
        model_name=model_name,
        system_instruction="You are a helpful assistant.",
    )
    call_args = mock_model.generate_content.call_args
    contents = call_args.kwargs["contents"]
    assert len(contents) == 3  # user -> assistant -> user
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "What is the capital of France?"
    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "Paris is the capital of France."
    assert contents[2].role == "user"
    assert contents[2].parts[0].text == "What about Germany?"


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_invoke_v2_no_system_message(GenerativeModelMock: MagicMock) -> None:
    """Test V2 interface invoke without system message."""
    model_name = "gemini-1.5-flash-001"
    messages: List[LLMMessage] = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    mock_response = Mock()
    mock_response.text = "I'm doing well, thank you!"
    mock_model = GenerativeModelMock.return_value
    mock_model.generate_content.return_value = mock_response

    llm = VertexAILLM(model_name=model_name)
    response = llm.invoke(messages)

    assert response.content == "I'm doing well, thank you!"
    GenerativeModelMock.assert_called_once_with(
        model_name=model_name,
        system_instruction=None,  # No system instruction should be used
    )


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
async def test_vertexai_ainvoke_v2_happy_path(GenerativeModelMock: MagicMock) -> None:
    """Test V2 interface async invoke method with List[LLMMessage] input."""
    model_name = "gemini-1.5-flash-001"
    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    mock_response = AsyncMock()
    mock_response.text = "2+2 equals 4."
    mock_model = GenerativeModelMock.return_value
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)

    llm = VertexAILLM(model_name=model_name)
    response = await llm.ainvoke(messages)

    assert response.content == "2+2 equals 4."
    GenerativeModelMock.assert_called_once_with(
        model_name=model_name,
        system_instruction="You are a helpful assistant.",
    )
    mock_model.generate_content_async.assert_awaited_once()


@patch("neo4j_graphrag.llm.vertexai_llm.VertexAILLM._parse_tool_response")
@patch("neo4j_graphrag.llm.vertexai_llm.VertexAILLM._call_brand_new_llm")
def test_vertexai_invoke_with_tools_v2(
    mock_call_llm: Mock,
    mock_parse_tool: Mock,
    test_tool: Tool,
) -> None:
    """Test V2 interface invoke_with_tools method with List[LLMMessage] input."""
    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather like?"},
    ]
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

    res = llm.invoke_with_tools(messages, tools)
    mock_call_llm.assert_called_once_with(
        messages,
        tools=tools,
    )
    mock_parse_tool.assert_called_once()
    assert isinstance(res, ToolCallResponse)


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.vertexai_llm.VertexAILLM._parse_tool_response")
@patch("neo4j_graphrag.llm.vertexai_llm.VertexAILLM._acall_brand_new_llm")
async def test_vertexai_ainvoke_with_tools_v2(
    mock_call_llm: Mock,
    mock_parse_tool: Mock,
    test_tool: Tool,
) -> None:
    """Test V2 interface async invoke_with_tools method with List[LLMMessage] input."""
    messages: List[LLMMessage] = [
        {"role": "user", "content": "What tools are available?"},
    ]
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

    res = await llm.ainvoke_with_tools(messages, tools)
    mock_call_llm.assert_awaited_once_with(
        messages,
        tools=tools,
    )
    mock_parse_tool.assert_called_once()
    assert isinstance(res, ToolCallResponse)


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_invoke_v2_validation_error(_GenerativeModelMock: MagicMock) -> None:
    """Test V2 interface invoke with invalid role raises error."""
    model_name = "gemini-1.5-flash-001"
    messages: List[LLMMessage] = [
        {"role": "invalid_role", "content": "This should fail."},  # type: ignore[typeddict-item]
    ]

    llm = VertexAILLM(model_name=model_name)

    with pytest.raises(ValueError) as exc_info:
        llm.invoke(messages)
    assert "Unknown role: invalid_role" in str(exc_info.value)


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_get_brand_new_messages_system_instruction_override(
    _GenerativeModelMock: MagicMock,
) -> None:
    """Test that system instruction in messages overrides class-level system instruction."""
    model_name = "gemini-1.5-flash-001"
    class_system_instruction = "You are a class-level assistant."
    messages: List[LLMMessage] = [
        {"role": "system", "content": "You are a message-level assistant."},
        {"role": "user", "content": "Hello"},
    ]

    llm = VertexAILLM(
        model_name=model_name, system_instruction=class_system_instruction
    )
    system_instruction, contents = llm.get_brand_new_messages(messages)

    assert system_instruction == "You are a message-level assistant."
    assert len(contents) == 1  # Only user message should remain
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Hello"
