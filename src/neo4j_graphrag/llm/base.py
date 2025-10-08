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

import warnings
from typing import Any, List, Optional, Sequence, Union

from pydantic import ValidationError

from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage

from .types import LLMResponse, ToolCallResponse
from neo4j_graphrag.utils.rate_limit import (
    DEFAULT_RATE_LIMIT_HANDLER,
    rate_limit_handler,
    async_rate_limit_handler,
    RateLimitHandler,
)

from neo4j_graphrag.tool import Tool

from .utils import legacy_inputs_to_messages
from ..exceptions import LLMGenerationError


class LLMInterface:
    """Interface for large language models.

    Args:
        model_name (str): The name of the language model.
        model_params (Optional[dict]): Additional parameters passed to the model when text is sent to it. Defaults to None.
        rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting. Defaults to retry with exponential backoff.
        **kwargs (Any): Arguments passed to the model when for the class is initialised. Defaults to None.
    """

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model_params = model_params or {}

        if rate_limit_handler is not None:
            self._rate_limit_handler = rate_limit_handler
        else:
            self._rate_limit_handler = DEFAULT_RATE_LIMIT_HANDLER

    @rate_limit_handler
    def invoke(
        self,
        input: Union[str, List[LLMMessage], MessageHistory],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        if message_history:
            warnings.warn(
                "Using 'message_history' in the llm.invoke method is deprecated. Please use invoke(list[LLMMessage]) instead.",
                DeprecationWarning,
            )
        if system_instruction:
            warnings.warn(
                "Using 'system_instruction' in the llm.invoke method is deprecated. Please use invoke(list[LLMMessage]) instead.",
                DeprecationWarning,
            )
        try:
            messages = legacy_inputs_to_messages(
                input, message_history, system_instruction
            )
        except ValidationError as e:
            raise LLMGenerationError("Input validation failed") from e
        return self._invoke(messages)

    def _invoke(
        self,
        input: list[LLMMessage],
    ) -> LLMResponse:
        """Sends a text input to the LLM and retrieves a response.

        Args:
            input (MessageHistory): Text sent to the LLM.

        Returns:
            LLMResponse: The response from the LLM.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        raise NotImplementedError()

    @async_rate_limit_handler
    async def ainvoke(
        self,
        input: Union[str, List[LLMMessage], MessageHistory],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        if message_history:
            warnings.warn(
                "Using 'message_history' in the llm.ainvoke method is deprecated. Please use invoke(list[LLMMessage]) instead.",
                DeprecationWarning,
            )
        if system_instruction:
            warnings.warn(
                "Using 'system_instruction' in the llm.ainvoke method is deprecated. Please use invoke(list[LLMMessage]) instead.",
                DeprecationWarning,
            )
        messages = legacy_inputs_to_messages(input, message_history, system_instruction)
        return await self._ainvoke(messages)

    async def _ainvoke(
        self,
        input: list[LLMMessage],
    ) -> LLMResponse:
        """Asynchronously sends a text input to the LLM and retrieves a response.

        Args:
            input (str): Text sent to the LLM.

        Returns:
            LLMResponse: The response from the LLM.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        raise NotImplementedError()

    @rate_limit_handler
    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Sends a text input to the LLM with tool definitions and retrieves a tool call response.

        This is a default implementation that should be overridden by LLM providers that support tool/function calling.

        Args:
            input (str): Text sent to the LLM.
            tools (Sequence[Tool]): Sequence of Tools for the LLM to choose from. Each LLM implementation should handle the conversion to its specific format.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            ToolCallResponse: The response from the LLM containing a tool call.

        Raises:
            LLMGenerationError: If anything goes wrong.
            NotImplementedError: If the LLM provider does not support tool calling.
        """
        try:
            messages = legacy_inputs_to_messages(
                input, message_history, system_instruction
            )
        except ValidationError as e:
            raise LLMGenerationError("Input validation failed") from e
        return self._invoke_with_tools(messages, tools)

    def _invoke_with_tools(
        self, inputs: list[LLMMessage], tools: Sequence[Tool]
    ) -> ToolCallResponse:
        raise NotImplementedError("This LLM provider does not support tool calling.")

    @async_rate_limit_handler
    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Asynchronously sends a text input to the LLM with tool definitions and retrieves a tool call response.

        This is a default implementation that should be overridden by LLM providers that support tool/function calling.

        Args:
            input (str): Text sent to the LLM.
            tools (Sequence[Tool]): Sequence of Tools for the LLM to choose from. Each LLM implementation should handle the conversion to its specific format.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            ToolCallResponse: The response from the LLM containing a tool call.

        Raises:
            LLMGenerationError: If anything goes wrong.
            NotImplementedError: If the LLM provider does not support tool calling.
        """
        messages = legacy_inputs_to_messages(input, message_history, system_instruction)
        return await self._ainvoke_with_tools(messages, tools)

    async def _ainvoke_with_tools(
        self, inputs: list[LLMMessage], tools: Sequence[Tool]
    ) -> ToolCallResponse:
        raise NotImplementedError("This LLM provider does not support tool calling.")
