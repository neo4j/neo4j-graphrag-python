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
from __future__ import annotations

from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import BedrockLLM
from neo4j_graphrag.llm.types import ToolCallResponse
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage


@pytest.fixture
def mock_boto3() -> Generator[MagicMock, None, None]:
    with patch("neo4j_graphrag.llm.bedrock_llm.boto3") as mock_boto:
        mock_client = MagicMock()
        mock_boto.client.return_value = mock_client
        yield mock_boto


def _make_converse_response(text: str = "generated text") -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        }
    }


def test_bedrock_llm_missing_dependency() -> None:
    with patch("neo4j_graphrag.llm.bedrock_llm.boto3", None):
        with pytest.raises(ImportError) as exc:
            BedrockLLM(model_name="us.anthropic.claude-sonnet-4-20250514-v1:0")
        assert "Could not import boto3 python client" in str(exc.value)


def test_bedrock_llm_default_model_from_env(mock_boto3: MagicMock) -> None:
    with patch.dict("os.environ", {"BEDROCK_LLM_MODEL": "custom-llm-model"}):
        import importlib
        import sys

        original_boto3 = sys.modules.get("boto3")
        sys.modules["boto3"] = mock_boto3

        try:
            import neo4j_graphrag.llm.bedrock_llm as bedrock_llm_mod

            importlib.reload(bedrock_llm_mod)

            assert bedrock_llm_mod.DEFAULT_BEDROCK_LLM_MODEL == "custom-llm-model"

            llm = bedrock_llm_mod.BedrockLLM()
            assert llm.model_name == "custom-llm-model"
        finally:
            if original_boto3 is not None:
                sys.modules["boto3"] = original_boto3
            importlib.reload(bedrock_llm_mod)


def test_bedrock_invoke_happy_path(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("hello world")

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    response = llm.invoke("hello")

    assert response.content == "hello world"
    mock_client.converse.assert_called_once()


def test_bedrock_invoke_with_message_history(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("response")

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    history: list[LLMMessage] = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]
    response = llm.invoke("follow up", message_history=history)

    assert response.content == "response"
    call_kwargs = mock_client.converse.call_args[1]
    # 2 history messages + 1 new user message
    assert len(call_kwargs["messages"]) == 3


def test_bedrock_invoke_with_system_instruction(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("response")

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    response = llm.invoke("hello", system_instruction="You are a bot")

    assert response.content == "response"
    call_kwargs = mock_client.converse.call_args[1]
    assert call_kwargs["system"] == [{"text": "You are a bot"}]


@pytest.mark.asyncio
async def test_bedrock_ainvoke_happy_path(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("async response")

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    response = await llm.ainvoke("hello")

    assert response.content == "async response"
    mock_client.converse.assert_called_once()


def test_bedrock_invoke_v2_happy_path(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("v2 response")

    messages: list[LLMMessage] = [
        {"role": "system", "content": "You are a bot"},
        {"role": "user", "content": "hello"},
    ]

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    response = llm.invoke(messages)

    assert response.content == "v2 response"
    call_kwargs = mock_client.converse.call_args[1]
    assert call_kwargs["system"] == [{"text": "You are a bot"}]
    # only user message, system is extracted
    assert len(call_kwargs["messages"]) == 1


@pytest.mark.asyncio
async def test_bedrock_ainvoke_v2_happy_path(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("async v2")

    messages: list[LLMMessage] = [{"role": "user", "content": "hello"}]

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    response = await llm.ainvoke(messages)

    assert response.content == "async v2"


def test_bedrock_invoke_error(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.side_effect = Exception("API error")

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    with pytest.raises(LLMGenerationError):
        llm.invoke("hello")


def test_bedrock_invoke_empty_response(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = {"output": {"message": {"content": []}}}

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    with pytest.raises(LLMGenerationError, match="LLM returned empty response"):
        llm.invoke("hello")


def test_bedrock_invoke_v2_with_response_format_raises_error(
    mock_boto3: MagicMock,
) -> None:
    messages: list[LLMMessage] = [{"role": "user", "content": "hello"}]
    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    with pytest.raises(NotImplementedError):
        llm.invoke(messages, response_format={"type": "json_object"})


def test_bedrock_invoke_with_model_params(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("response")

    llm = BedrockLLM(
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        model_params={"temperature": 0.5, "maxTokens": 512},
    )
    llm.invoke("hello")

    call_kwargs = mock_client.converse.call_args[1]
    assert call_kwargs["inferenceConfig"] == {"temperature": 0.5, "maxTokens": 512}


def test_bedrock_invoke_with_region_name(mock_boto3: MagicMock) -> None:
    BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1")
    mock_boto3.client.assert_called_once_with(
        "bedrock-runtime", region_name="us-east-1"
    )


def test_bedrock_invoke_v2_error(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.side_effect = Exception("API error")
    messages: list[LLMMessage] = [{"role": "user", "content": "hello"}]
    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    with pytest.raises(LLMGenerationError):
        llm.invoke(messages)


def test_bedrock_parse_response_with_usage(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = {
        **_make_converse_response("hello"),
        "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
    }
    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    response = llm.invoke("hello")
    assert response.usage is not None
    assert response.usage.request_tokens == 10
    assert response.usage.response_tokens == 20
    assert response.usage.total_tokens == 30


def test_bedrock_invoke_with_tools_happy_path(
    mock_boto3: MagicMock, test_tool: Tool
) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "name": "test_tool",
                            "input": {"param1": "value1"},
                        }
                    }
                ]
            }
        }
    }
    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    res = llm.invoke_with_tools("hello", [test_tool])
    assert isinstance(res, ToolCallResponse)
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "test_tool"
    assert res.tool_calls[0].arguments == {"param1": "value1"}


def test_bedrock_invoke_with_tools_error(
    mock_boto3: MagicMock, test_tool: Tool
) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.side_effect = Exception("API error")
    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    with pytest.raises(LLMGenerationError):
        llm.invoke_with_tools("hello", [test_tool])


@pytest.mark.asyncio
async def test_bedrock_ainvoke_error(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.side_effect = Exception("API error")
    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    with pytest.raises(LLMGenerationError):
        await llm.ainvoke("hello")


@pytest.mark.asyncio
async def test_bedrock_ainvoke_v2_with_response_format_raises_error(
    mock_boto3: MagicMock,
) -> None:
    messages: list[LLMMessage] = [{"role": "user", "content": "hello"}]
    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    with pytest.raises(NotImplementedError):
        await llm.ainvoke(messages, response_format={"type": "json_object"})


@pytest.mark.asyncio
async def test_bedrock_ainvoke_v2_error(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.side_effect = Exception("API error")
    messages: list[LLMMessage] = [{"role": "user", "content": "hello"}]
    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    with pytest.raises(LLMGenerationError):
        await llm.ainvoke(messages)


@pytest.mark.asyncio
async def test_bedrock_ainvoke_with_tools_happy_path(
    mock_boto3: MagicMock, test_tool: Tool
) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "name": "test_tool",
                            "input": {"param1": "value1"},
                        }
                    }
                ]
            }
        }
    }
    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    res = await llm.ainvoke_with_tools("hello", [test_tool])
    assert isinstance(res, ToolCallResponse)
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "test_tool"


@pytest.mark.asyncio
async def test_bedrock_ainvoke_with_tools_error(
    mock_boto3: MagicMock, test_tool: Tool
) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.side_effect = Exception("API error")
    llm = BedrockLLM("us.anthropic.claude-sonnet-4-20250514-v1:0")
    with pytest.raises(LLMGenerationError):
        await llm.ainvoke_with_tools("hello", [test_tool])
