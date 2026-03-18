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
from tenacity import RetryError

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.bedrock_llm import BedrockLLM
from neo4j_graphrag.llm.types import ToolCallResponse
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage


def get_mock_converse_response(content: str = "bedrock response") -> dict[str, Any]:
    """Create a mock Converse API response."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": content}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 20},
    }


def get_mock_tool_response(
    tool_name: str = "test_tool",
    tool_input: dict[str, Any] | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    """Create a mock Converse API response with tool use."""
    content_blocks = []
    if content:
        content_blocks.append({"text": content})
    content_blocks.append({
        "toolUse": {
            "name": tool_name,
            "toolUseId": "tool-123",
            "input": tool_input or {"param1": "value1"},
        }
    })
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": content_blocks,
            }
        },
        "stopReason": "tool_use",
    }


@patch("neo4j_graphrag.llm.bedrock_llm.boto3", None)
def test_bedrock_llm_missing_dependency() -> None:
    with pytest.raises(ImportError):
        BedrockLLM()


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_happy_path(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_converse_response("Hello from Bedrock!")

    llm = BedrockLLM(region_name="us-east-1")
    res = llm.invoke("What is graph RAG?")

    assert isinstance(res, LLMResponse)
    assert res.content == "Hello from Bedrock!"

    mock_boto3.client.assert_called_once_with(
        "bedrock-runtime",
        region_name="us-east-1",
    )


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_default_model(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_converse_response()

    llm = BedrockLLM()
    llm.invoke("test")

    call_args = mock_client.converse.call_args
    assert call_args.kwargs["modelId"] == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_custom_model(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_converse_response()

    llm = BedrockLLM(model_id="anthropic.claude-3-5-haiku-20241022-v1:0")
    llm.invoke("test")

    call_args = mock_client.converse.call_args
    assert call_args.kwargs["modelId"] == "anthropic.claude-3-5-haiku-20241022-v1:0"


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_inference_profile(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_converse_response()

    inference_profile_arn = (
        "arn:aws:bedrock:us-east-1:123456789:inference-profile/my-profile"
    )
    llm = BedrockLLM(inference_profile_id=inference_profile_arn)
    llm.invoke("test")

    call_args = mock_client.converse.call_args
    assert call_args.kwargs["modelId"] == inference_profile_arn


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_with_preconfigured_client(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_client.converse.return_value = get_mock_converse_response("custom client")

    llm = BedrockLLM(client=mock_client)
    res = llm.invoke("test")

    assert res.content == "custom client"
    mock_boto3.client.assert_not_called()


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_with_model_params(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_converse_response()

    model_params = {"temperature": 0.7, "maxTokens": 1000, "topP": 0.9}
    llm = BedrockLLM(model_params=model_params)
    llm.invoke("test")

    call_args = mock_client.converse.call_args
    assert call_args.kwargs["inferenceConfig"] == model_params


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_with_system_instruction(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_converse_response()

    llm = BedrockLLM()
    llm.invoke("What is 2+2?", system_instruction="You are a math tutor.")

    call_args = mock_client.converse.call_args
    assert call_args.kwargs["system"] == [{"text": "You are a math tutor."}]


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_with_message_history(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_converse_response()

    llm = BedrockLLM()
    message_history = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]
    llm.invoke("And Germany?", message_history=message_history)  # type: ignore

    call_args = mock_client.converse.call_args
    messages = call_args.kwargs["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"text": "What is the capital of France?"}]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == [{"text": "Paris."}]
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == [{"text": "And Germany?"}]


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_error_handling(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.side_effect = Exception("ValidationException: Invalid input")

    llm = BedrockLLM()

    with pytest.raises(LLMGenerationError, match="Failed to generate text with Bedrock"):
        llm.invoke("test")


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_empty_response(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = {"output": {"message": {"content": []}}}

    llm = BedrockLLM()

    with pytest.raises(LLMGenerationError, match="LLM returned empty response"):
        llm.invoke("test")


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_rate_limit_retries(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    mock_client.converse.side_effect = [
        Exception("ThrottlingException: rate limit exceeded"),
        Exception("ThrottlingException: rate limit exceeded"),
        Exception("ThrottlingException: rate limit exceeded"),
    ]

    llm = BedrockLLM()

    with pytest.raises(RetryError):
        llm.invoke("test")

    assert mock_client.converse.call_count == 3


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_rate_limit_eventual_success(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    mock_client.converse.side_effect = [
        Exception("rate limit exceeded"),
        Exception("rate limit exceeded"),
        get_mock_converse_response("success after retries"),
    ]

    llm = BedrockLLM()
    result = llm.invoke("test")

    assert result.content == "success after retries"
    assert mock_client.converse.call_count == 3


# V2 Interface Tests
@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_invoke_v2_happy_path(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_converse_response("v2 response")

    messages: list[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is graph RAG?"},
    ]

    llm = BedrockLLM()
    res = llm.invoke(messages)

    assert isinstance(res, LLMResponse)
    assert res.content == "v2 response"

    call_args = mock_client.converse.call_args
    assert call_args.kwargs["system"] == [{"text": "You are a helpful assistant."}]
    assert len(call_args.kwargs["messages"]) == 1
    assert call_args.kwargs["messages"][0]["role"] == "user"


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_invoke_v2_multi_turn(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_converse_response("multi-turn response")

    messages: list[LLMMessage] = [
        {"role": "user", "content": "What is Neo4j?"},
        {"role": "assistant", "content": "Neo4j is a graph database."},
        {"role": "user", "content": "How does it work?"},
    ]

    llm = BedrockLLM()
    res = llm.invoke(messages)

    assert res.content == "multi-turn response"

    call_args = mock_client.converse.call_args
    assert len(call_args.kwargs["messages"]) == 3


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_invoke_v2_unknown_role(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    messages: list[LLMMessage] = [
        {"role": "unknown", "content": "test"},  # type: ignore
    ]

    llm = BedrockLLM()

    with pytest.raises(LLMGenerationError):
        llm.invoke(messages)


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_invalid_input_type(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    llm = BedrockLLM()

    with pytest.raises(ValueError, match="Invalid input type for invoke method"):
        llm.invoke(123)  # type: ignore


# Async Tests
@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
async def test_bedrock_llm_ainvoke_v1_happy_path(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_converse_response("async response")

    llm = BedrockLLM()
    res = await llm.ainvoke("What is graph RAG?")

    assert isinstance(res, LLMResponse)
    assert res.content == "async response"


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
async def test_bedrock_llm_ainvoke_v2_happy_path(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_converse_response("async v2 response")

    messages: list[LLMMessage] = [
        {"role": "user", "content": "Hello!"},
    ]

    llm = BedrockLLM()
    res = await llm.ainvoke(messages)

    assert isinstance(res, LLMResponse)
    assert res.content == "async v2 response"


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
async def test_bedrock_llm_ainvoke_invalid_input_type(mock_boto3: Mock) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    llm = BedrockLLM()

    with pytest.raises(ValueError, match="Invalid input type for ainvoke method"):
        await llm.ainvoke({"invalid": "dict"})  # type: ignore


# Tool Calling Tests
@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_invoke_with_tools_happy_path(
    mock_boto3: Mock,
    test_tool: Tool,
) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_tool_response()

    llm = BedrockLLM()
    res = llm.invoke_with_tools("Call the test tool", [test_tool])

    assert isinstance(res, ToolCallResponse)
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "test_tool"
    assert res.tool_calls[0].arguments == {"param1": "value1"}


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_invoke_with_tools_with_content(
    mock_boto3: Mock,
    test_tool: Tool,
) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_tool_response(
        content="I'll call the tool for you"
    )

    llm = BedrockLLM()
    res = llm.invoke_with_tools("Call the test tool", [test_tool])

    assert isinstance(res, ToolCallResponse)
    assert res.content == "I'll call the tool for you"
    assert len(res.tool_calls) == 1


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_invoke_with_tools_no_tool_call(
    mock_boto3: Mock,
    test_tool: Tool,
) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_converse_response(
        "I don't need to use any tools"
    )

    llm = BedrockLLM()
    res = llm.invoke_with_tools("What is 2+2?", [test_tool])

    assert isinstance(res, ToolCallResponse)
    assert len(res.tool_calls) == 0
    assert res.content == "I don't need to use any tools"


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_invoke_with_tools_format(
    mock_boto3: Mock,
    test_tool: Tool,
) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_tool_response()

    llm = BedrockLLM()
    llm.invoke_with_tools("test", [test_tool])

    call_args = mock_client.converse.call_args
    tool_config = call_args.kwargs["toolConfig"]

    assert "tools" in tool_config
    assert len(tool_config["tools"]) == 1
    assert tool_config["tools"][0]["toolSpec"]["name"] == "test_tool"
    assert tool_config["tools"][0]["toolSpec"]["description"] == "A test tool"
    assert "inputSchema" in tool_config["tools"][0]["toolSpec"]


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_invoke_with_tools_error(
    mock_boto3: Mock,
    test_tool: Tool,
) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.side_effect = Exception("Tool error")

    llm = BedrockLLM()

    with pytest.raises(LLMGenerationError, match="Failed to invoke with tools"):
        llm.invoke_with_tools("test", [test_tool])


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
async def test_bedrock_llm_ainvoke_with_tools_happy_path(
    mock_boto3: Mock,
    test_tool: Tool,
) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_tool_response()

    llm = BedrockLLM()
    res = await llm.ainvoke_with_tools("Call the test tool", [test_tool])

    assert isinstance(res, ToolCallResponse)
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "test_tool"


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_invoke_with_tools_with_system_instruction(
    mock_boto3: Mock,
    test_tool: Tool,
) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_tool_response()

    llm = BedrockLLM()
    llm.invoke_with_tools(
        "test",
        [test_tool],
        system_instruction="You are a helpful assistant.",
    )

    call_args = mock_client.converse.call_args
    assert call_args.kwargs["system"] == [{"text": "You are a helpful assistant."}]


@patch("neo4j_graphrag.llm.bedrock_llm.boto3")
def test_bedrock_llm_invoke_with_tools_with_message_history(
    mock_boto3: Mock,
    test_tool: Tool,
) -> None:
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.converse.return_value = get_mock_tool_response()

    llm = BedrockLLM()
    message_history = [
        {"role": "user", "content": "What tools do you have?"},
        {"role": "assistant", "content": "I have a test tool."},
    ]
    llm.invoke_with_tools("Use it", [test_tool], message_history=message_history)  # type: ignore

    call_args = mock_client.converse.call_args
    messages = call_args.kwargs["messages"]
    assert len(messages) == 3
