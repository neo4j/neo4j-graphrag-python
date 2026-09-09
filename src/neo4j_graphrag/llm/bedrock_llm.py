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

import asyncio
import os
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

# 3rd party dependencies
from pydantic import BaseModel, ValidationError

# project dependencies
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMBase
from neo4j_graphrag.llm.types import (
    BaseMessage,
    LLMResponse,
    LLMUsage,
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

DEFAULT_BEDROCK_LLM_MODEL = os.getenv(
    "BEDROCK_LLM_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)


# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from, no-else-return, import-outside-toplevel
class BedrockLLM(LLMBase):
    """LLM interface for Amazon Bedrock via the boto3 Converse API.

    Args:
        model_name (str): Bedrock model ID. Defaults to the ``BEDROCK_LLM_MODEL``
            environment variable, or "us.anthropic.claude-haiku-4-5-20251001-v1:0" if not set.
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
            model_name="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            model_params={"temperature": 0.7, "maxTokens": 1024},
            region_name="us-east-1",
        )
        llm.invoke("Who is the mother of Paul Atreides?")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BEDROCK_LLM_MODEL,
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
        LLMBase.__init__(
            self,
            model_name=model_name,
            model_params=model_params or {},
            rate_limit_handler=rate_limit_handler,
        )
        client_kwargs: dict[str, Any] = {**kwargs}
        if region_name:
            client_kwargs["region_name"] = region_name
        self.client = boto3.client("bedrock-runtime", **client_kwargs)

    # implementations
    def _build_v1_request(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build the ``client.converse`` kwargs for a v1 (str-input) call."""
        messages = self.get_messages(input, message_history)
        return self._build_converse_kwargs(
            messages, system_instruction=system_instruction
        )

    def _build_v2_request(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the ``client.converse`` kwargs for a v2 (message-list) call."""
        if response_format is not None:
            raise NotImplementedError(
                "BedrockLLM does not currently support structured output"
            )
        system_instruction, messages = self.get_messages_v2(input)
        return self._build_converse_kwargs(
            messages, system_instruction=system_instruction, **kwargs
        )

    def _call_sync(self, converse_kwargs: dict[str, Any]) -> LLMResponse:
        """Sync transport hook: the only place ``client.converse`` is called."""
        try:
            response = self.client.converse(**converse_kwargs)
            return self._parse_response(response)
        except Exception as e:
            raise LLMGenerationError(f"Error calling BedrockLLM: {e}") from e

    async def _call_async(self, converse_kwargs: dict[str, Any]) -> LLMResponse:
        """Async transport hook: boto3 is sync-only, so the same
        ``client.converse`` call runs in a thread-pool executor.

        Only this method carries the async retry layer (via
        :func:`async_rate_limit_handler` on ``_ainvoke_v1``/``_ainvoke_v2``);
        it dispatches to :meth:`_call_sync` directly rather than to a
        rate-limited sync entry point, so a single call never gets retried
        by both the sync and async decorators at once.
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._call_sync, converse_kwargs)
        except LLMGenerationError:
            raise
        except Exception as e:
            raise LLMGenerationError(f"Error calling BedrockLLM: {e}") from e

    @rate_limit_handler_decorator
    def _invoke_v1(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        request = self._build_v1_request(input, message_history, system_instruction)
        return self._call_sync(request)

    @rate_limit_handler_decorator
    def _invoke_v2(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        request = self._build_v2_request(
            input, response_format=response_format, **kwargs
        )
        return self._call_sync(request)

    @async_rate_limit_handler_decorator
    async def _ainvoke_v1(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        request = self._build_v1_request(input, message_history, system_instruction)
        return await self._call_async(request)

    @async_rate_limit_handler_decorator
    async def _ainvoke_v2(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        request = self._build_v2_request(
            input, response_format=response_format, **kwargs
        )
        return await self._call_async(request)

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
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.invoke_with_tools,
                input,
                tools,
                message_history,
                system_instruction,
            )
        except LLMGenerationError:
            raise
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
        usage = None
        raw_usage = response.get("usage", {})
        if raw_usage:
            usage = LLMUsage(
                request_tokens=raw_usage.get("inputTokens"),
                response_tokens=raw_usage.get("outputTokens"),
                total_tokens=raw_usage.get("totalTokens"),
            )
        return LLMResponse(content="".join(text_parts), usage=usage)

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
