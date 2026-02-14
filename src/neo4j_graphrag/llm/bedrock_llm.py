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

# built-in dependencies
from __future__ import annotations

from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    overload,
)

# 3rd party dependencies
from pydantic import BaseModel, ValidationError

# project dependencies
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMInterface, LLMInterfaceV2
from neo4j_graphrag.llm.types import (
    BaseMessage,
    LLMResponse,
    MessageList,
    ToolCall,
    ToolCallResponse,
)
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
)
from neo4j_graphrag.utils.rate_limit import (
    async_rate_limit_handler as async_rate_limit_handler_decorator,
)
from neo4j_graphrag.utils.rate_limit import (
    rate_limit_handler as rate_limit_handler_decorator,
)

try:
    import boto3
except ImportError:
    boto3 = None


# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from, no-else-return, import-outside-toplevel
class BedrockLLM(LLMInterface, LLMInterfaceV2):
    """LLM interface for Amazon Bedrock via the boto3 Converse API.

    Args:
        model_name (str): Bedrock model ID. Defaults to "us.anthropic.claude-sonnet-4-20250514-v1:0".
        model_params (Optional[dict]): Additional parameters passed to the model
            (e.g. ``{"temperature": 0.7, "maxTokens": 1024}``).
        region_name (Optional[str]): AWS region. Defaults to boto3 session default.
        rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting.
        **kwargs (Any): Arguments passed to ``boto3.client("bedrock-runtime", ...)``.

    Raises:
        LLMGenerationError: If there's an error generating the response from the model.

    Example:

    .. code-block:: python

        from neo4j_graphrag.llm import BedrockLLM

        llm = BedrockLLM(
            model_name="us.anthropic.claude-sonnet-4-20250514-v1:0",
            model_params={"temperature": 0.7, "maxTokens": 1024},
            region_name="us-east-1",
        )
        llm.invoke("Who is the mother of Paul Atreides?")
    """

    def __init__(
        self,
        model_name: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
        model_params: Optional[dict[str, Any]] = None,
        region_name: Optional[str] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ) -> None:
        if boto3 is None:
            raise ImportError(
                "Could not import boto3 python client. "
                'Please install it with `pip install "neo4j-graphrag[bedrock]"`.'
            )
        LLMInterfaceV2.__init__(
            self,
            model_name=model_name,
            model_params=model_params or {},
            rate_limit_handler=rate_limit_handler,
        )
        client_kwargs: dict[str, Any] = {**kwargs}
        if region_name:
            client_kwargs["region_name"] = region_name
        self.client = boto3.client("bedrock-runtime", **client_kwargs)

    # overloads for LLMInterface and LLMInterfaceV2 methods
    @overload  # type: ignore[no-overload-impl]
    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse: ...

    @overload
    def invoke(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    @overload  # type: ignore[no-overload-impl]
    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse: ...

    @overload
    async def ainvoke(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    # switching logic
    def invoke(  # type: ignore[no-redef]
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return self.__invoke_v1(input, message_history, system_instruction)
        return self.__invoke_v2(input, response_format=response_format, **kwargs)

    async def ainvoke(  # type: ignore[no-redef]
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return await self.__ainvoke_v1(input, message_history, system_instruction)
        return await self.__ainvoke_v2(input, response_format=response_format, **kwargs)

    # implementations
    @rate_limit_handler_decorator
    def __invoke_v1(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        try:
            messages = self.get_messages(input, message_history)
            converse_kwargs = self._build_converse_kwargs(
                messages, system_instruction=system_instruction
            )
            response = self.client.converse(**converse_kwargs)
            return self._parse_response(response)
        except Exception as e:
            raise LLMGenerationError(f"Error calling BedrockLLM: {e}") from e

    @rate_limit_handler_decorator
    def __invoke_v2(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if response_format is not None:
            raise NotImplementedError(
                "BedrockLLM does not currently support structured output"
            )
        try:
            system_instruction, messages = self.get_messages_v2(input)
            converse_kwargs = self._build_converse_kwargs(
                messages, system_instruction=system_instruction, **kwargs
            )
            response = self.client.converse(**converse_kwargs)
            return self._parse_response(response)
        except Exception as e:
            raise LLMGenerationError(f"Error calling BedrockLLM: {e}") from e

    @async_rate_limit_handler_decorator
    async def __ainvoke_v1(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        # boto3 does not have native async support; run synchronously
        try:
            messages = self.get_messages(input, message_history)
            converse_kwargs = self._build_converse_kwargs(
                messages, system_instruction=system_instruction
            )
            response = self.client.converse(**converse_kwargs)
            return self._parse_response(response)
        except Exception as e:
            raise LLMGenerationError(f"Error calling BedrockLLM: {e}") from e

    @async_rate_limit_handler_decorator
    async def __ainvoke_v2(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if response_format is not None:
            raise NotImplementedError(
                "BedrockLLM does not currently support structured output"
            )
        # boto3 does not have native async support; run synchronously
        try:
            system_instruction, messages = self.get_messages_v2(input)
            converse_kwargs = self._build_converse_kwargs(
                messages, system_instruction=system_instruction, **kwargs
            )
            response = self.client.converse(**converse_kwargs)
            return self._parse_response(response)
        except Exception as e:
            raise LLMGenerationError(f"Error calling BedrockLLM: {e}") from e

    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        try:
            messages = self.get_messages(input, message_history)
            tool_config = self._get_tool_config(tools)
            converse_kwargs = self._build_converse_kwargs(
                messages,
                system_instruction=system_instruction,
                toolConfig=tool_config,
            )
            response = self.client.converse(**converse_kwargs)
            return self._parse_tool_response(response)
        except Exception as e:
            raise LLMGenerationError(f"Error calling BedrockLLM with tools: {e}") from e

    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        try:
            messages = self.get_messages(input, message_history)
            tool_config = self._get_tool_config(tools)
            converse_kwargs = self._build_converse_kwargs(
                messages,
                system_instruction=system_instruction,
                toolConfig=tool_config,
            )
            response = self.client.converse(**converse_kwargs)
            return self._parse_tool_response(response)
        except Exception as e:
            raise LLMGenerationError(f"Error calling BedrockLLM with tools: {e}") from e

    # subsidiary methods
    def get_messages(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
    ) -> list[dict[str, Any]]:
        """Constructs the message list for the Bedrock Converse API."""
        messages: list[dict[str, Any]] = []
        if message_history:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            try:
                MessageList(messages=cast(list[BaseMessage], message_history))
            except ValidationError as e:
                raise LLMGenerationError(e.errors()) from e

            for message in message_history:
                role = message.get("role")
                content = message.get("content", "")
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": [{"text": content}]})

        messages.append({"role": "user", "content": [{"text": input}]})
        return messages

    def get_messages_v2(
        self,
        input: list[LLMMessage],
    ) -> tuple[Optional[str], list[dict[str, Any]]]:
        """Constructs the message list for the Bedrock Converse API from V2 input."""
        messages: list[dict[str, Any]] = []
        system_instruction: Optional[str] = None
        for message in input:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_instruction = content
            elif role in ("user", "assistant"):
                messages.append({"role": role, "content": [{"text": content}]})
        return system_instruction, messages

    def _build_converse_kwargs(
        self,
        messages: list[dict[str, Any]],
        system_instruction: Optional[str] = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Builds the kwargs dict for the Bedrock converse() call."""
        kwargs: dict[str, Any] = {
            "modelId": self.model_name,
            "messages": messages,
        }
        if system_instruction:
            kwargs["system"] = [{"text": system_instruction}]

        # merge model_params into inferenceConfig
        if self.model_params:
            kwargs["inferenceConfig"] = {**self.model_params}

        kwargs.update(extra)
        return kwargs

    def _parse_response(self, response: dict[str, Any]) -> LLMResponse:
        """Extracts text content from a Bedrock converse() response."""
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        text_parts = [block["text"] for block in content_blocks if "text" in block]
        if not text_parts:
            raise LLMGenerationError("LLM returned empty response.")
        return LLMResponse(content="".join(text_parts))

    def _get_tool_config(
        self, tools: Optional[Sequence[Tool]]
    ) -> Optional[dict[str, Any]]:
        """Converts Tool objects to Bedrock toolConfig format."""
        if not tools:
            return None
        tool_defs = []
        for tool in tools:
            tool_defs.append(
                {
                    "toolSpec": {
                        "name": tool.get_name(),
                        "description": tool.get_description(),
                        "inputSchema": {
                            "json": tool.get_parameters(
                                exclude=["additional_properties"]
                            )
                        },
                    }
                }
            )
        return {"tools": tool_defs}

    def _parse_tool_response(self, response: dict[str, Any]) -> ToolCallResponse:
        """Extracts tool calls from a Bedrock converse() response."""
        tool_calls: list[ToolCall] = []
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        for block in content_blocks:
            if "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        name=tool_use.get("name", ""),
                        arguments=tool_use.get("input", {}),
                    )
                )
        return ToolCallResponse(tool_calls=tool_calls, content=None)
