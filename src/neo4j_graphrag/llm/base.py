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

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Type, Union

from pydantic import BaseModel

from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.rate_limit import (
    DEFAULT_RATE_LIMIT_HANDLER,
    RateLimitHandler,
)
from neo4j_graphrag.utils.rate_limit import (
    async_rate_limit_handler as async_rate_limit_handler_decorator,
)
from neo4j_graphrag.utils.rate_limit import (
    rate_limit_handler as rate_limit_handler_decorator,
)

from .types import LLMResponse, ToolCallResponse

# pylint: disable=redefined-builtin

logger = logging.getLogger(__name__)


class _LLMConfigMixin:
    """Shared configuration state for both the sync and async LLM interfaces.

    Owning __init__ here (rather than on SyncLLMInterface or AsyncLLMInterface
    individually) means either interface is fully self-sufficient standalone,
    with no reliance on MRO order to pick up model_name/model_params/
    rate_limit_handler setup.

    Args:
        model_name (str): The name of the language model.
        model_params (Optional[dict]): Additional parameters passed to the model when text is sent to it. Defaults to None.
        rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting. Defaults to retry with exponential backoff.
        **kwargs (Any): Arguments passed to the model when for the class is initialised. Defaults to None.
    """

    supports_structured_output: bool = False
    """Whether this LLM supports structured output (response_format with Pydantic models or json schema)."""

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.model_params = model_params or {}

        if rate_limit_handler is not None:
            self._rate_limit_handler = rate_limit_handler
        else:
            self._rate_limit_handler = DEFAULT_RATE_LIMIT_HANDLER


class SyncLLMInterface(_LLMConfigMixin, ABC):
    """Synchronous half of the LLM contract: invoke() plus the hooks it needs."""

    def invoke(
        self,
        input: List[LLMMessage],
        *,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Sends a list of messages to the LLM and retrieves a response.

        Args:
            input (List[LLMMessage]): Messages sent to the LLM.
            response_format (Optional[Union[Type[BaseModel], dict[str, Any]]]): Optional
                response format specification. Can be a Pydantic model class for structured
                output or a dict for provider-specific formats. Defaults to None.

        Returns:
            LLMResponse: The response from the LLM.

        Raises:
            LLMGenerationError: If anything goes wrong.
            NotImplementedError: If the LLM provider does not support structured output.
        """
        request = self._build_request(input, response_format=response_format, **kwargs)
        raw_response = self._call_sync_with_rate_limit(request)
        return self._parse_response(raw_response)

    @rate_limit_handler_decorator
    def _call_sync_with_rate_limit(self, request: Any) -> Any:
        return self._call_sync(request)

    @abstractmethod
    def _build_request(
        self,
        messages: List[LLMMessage],
        *,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Build a provider-specific request from the common message/response-format input.

        Implementations must raise LLMGenerationError for any build-time failure
        (message validation, unsupported response_format, schema conversion errors)
        so callers see one consistent exception type for the whole invoke/ainvoke call.
        """

    @abstractmethod
    def _parse_response(self, raw_response: Any) -> LLMResponse:
        """Parse a provider-specific raw response into the common LLMResponse shape."""

    @abstractmethod
    def _call_sync(self, request: Any) -> Any:
        """Send the built request to the LLM synchronously and return the raw response.

        Implementations must wrap SDK/transport exceptions into LLMGenerationError.
        """

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
        raise NotImplementedError("This LLM provider does not support tool calling.")


class AsyncLLMInterface(_LLMConfigMixin, ABC):
    """Asynchronous half of the LLM contract: ainvoke() plus the hooks it needs."""

    async def ainvoke(
        self,
        input: List[LLMMessage],
        *,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Asynchronously sends a list of messages to the LLM and retrieves a response.

        Args:
            input (List[LLMMessage]): Messages sent to the LLM.
            response_format (Optional[Union[Type[BaseModel], dict[str, Any]]]): Optional
                response format specification. Can be a Pydantic model class for structured
                output or a dict for provider-specific formats. Defaults to None.

        Returns:
            LLMResponse: The response from the LLM.

        Raises:
            LLMGenerationError: If anything goes wrong.
            NotImplementedError: If the LLM provider does not support structured output.
        """
        request = self._build_request(input, response_format=response_format, **kwargs)
        raw_response = await self._call_async_with_rate_limit(request)
        return self._parse_response(raw_response)

    @async_rate_limit_handler_decorator
    async def _call_async_with_rate_limit(self, request: Any) -> Any:
        return await self._call_async(request)

    @abstractmethod
    def _build_request(
        self,
        messages: List[LLMMessage],
        *,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Build a provider-specific request from the common message/response-format input.

        Implementations must raise LLMGenerationError for any build-time failure
        (message validation, unsupported response_format, schema conversion errors)
        so callers see one consistent exception type for the whole invoke/ainvoke call.
        """

    @abstractmethod
    def _parse_response(self, raw_response: Any) -> LLMResponse:
        """Parse a provider-specific raw response into the common LLMResponse shape."""

    @abstractmethod
    async def _call_async(self, request: Any) -> Any:
        """Send the built request to the LLM asynchronously and return the raw response.

        Implementations must wrap SDK/transport exceptions into LLMGenerationError.
        """

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
        raise NotImplementedError("This LLM provider does not support tool calling.")


class LLMBase(SyncLLMInterface, AsyncLLMInterface, ABC):
    """Combined sync+async LLM contract every provider implements.

    _LLMConfigMixin.__init__ is resolved exactly once via C3 linearization
    (it is the shared leaf ancestor of both SyncLLMInterface and
    AsyncLLMInterface), so this combination does not double-initialize.
    """

    def close(self) -> None:
        """Close both clients and release any resources.

        Must not be called from a running async context — use ``await aclose()`` instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop: safe to block
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.aclose())
            finally:
                loop.close()
        else:
            raise RuntimeError(
                "Cannot call close() from a running async context. "
                "Use 'async with' or 'await aclose()' instead."
            )

    async def aclose(self) -> None:
        """Close both clients and release any resources.

        Override in subclasses that hold HTTP clients.
        """
        pass

    def __enter__(self) -> "LLMBase":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "LLMBase":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()
