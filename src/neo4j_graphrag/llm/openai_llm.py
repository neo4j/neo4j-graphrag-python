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

import abc
import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
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

from ..exceptions import LLMGenerationError
from .base import LLMBase
from .types import (
    BaseMessage,
    LLMResponse,
    LLMUsage,
    MessageList,
    SystemMessage,
    ToolCall,
    ToolCallResponse,
    UserMessage,
)
from .utils import split_http_client_kwargs

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionToolParam,
    )
else:
    ChatCompletionMessageParam = Any
    ChatCompletionToolParam = Any
    OpenAI = Any
    AsyncOpenAI = Any

logger = logging.getLogger(__name__)


# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from, no-else-return, import-outside-toplevel, line-too-long
class BaseOpenAILLM(LLMBase, abc.ABC):
    """Base class for OpenAI LLMs."""

    client: OpenAI
    async_client: AsyncOpenAI

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ):
        """
        Base class for OpenAI LLM.

        Makes sure the openai Python client is installed during init.

        Args:
            model_name (str):
            model_params (str): Parameters like temperature that will be passed to the model when text is sent to it. Defaults to None.
            rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting. Defaults to retry with exponential backoff.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                """Could not import openai Python client.
                Please install it with `pip install "neo4j-graphrag[openai]"`."""
            )
        self.openai = openai

        LLMBase.__init__(
            self,
            model_name=model_name,
            model_params=model_params or {},
            rate_limit_handler=rate_limit_handler,
            **kwargs,
        )

    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],  # Tools definition as a sequence of Tool objects
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        return self.__invoke_v1_with_tools(
            input, tools, message_history, system_instruction
        )

    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        return await self.__ainvoke_v1_with_tools(
            input, tools, message_history, system_instruction
        )

    async def aclose(self) -> None:
        self.client.close()
        await self.async_client.close()

    # subsidiary methods
    def get_messages(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> Iterable[ChatCompletionMessageParam]:
        """Constructs the message list for OpenAI chat completion for legacy LLMInterface."""
        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction).model_dump())
        if message_history:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            try:
                MessageList(messages=cast(list[BaseMessage], message_history))
            except ValidationError as e:
                raise LLMGenerationError(e.errors()) from e
            messages.extend(cast(Iterable[dict[str, Any]], message_history))
        messages.append(UserMessage(content=input).model_dump())
        return messages  # type: ignore

    def get_messages_v2(
        self,
        messages: list[LLMMessage],
    ) -> Iterable[ChatCompletionMessageParam]:
        """Constructs the message list for OpenAI chat completion for LLMInterfaceV2."""
        chat_messages = []
        for m in messages:
            message_type: Type[ChatCompletionMessageParam]
            if m["role"] == "system":
                message_type = self.openai.types.chat.ChatCompletionSystemMessageParam
            elif m["role"] == "user":
                message_type = self.openai.types.chat.ChatCompletionUserMessageParam
            elif m["role"] == "assistant":
                message_type = (
                    self.openai.types.chat.ChatCompletionAssistantMessageParam
                )
            else:
                raise ValueError(f"Unknown role: {m['role']}")
            chat_messages.append(
                message_type(
                    role=m["role"],  # type: ignore
                    content=m["content"],
                )
            )
        return chat_messages

    def _convert_tool_to_openai_format(self, tool: Tool) -> Dict[str, Any]:
        """Convert a Tool object to OpenAI's expected format.

        Args:
            tool: A Tool object to convert to OpenAI's format.

        Returns:
            A dictionary in OpenAI's tool format.
        """
        try:
            return {
                "type": "function",
                "function": {
                    "name": tool.get_name(),
                    "description": tool.get_description(),
                    "parameters": tool.get_parameters(),
                },
            }
        except AttributeError:
            raise LLMGenerationError(f"Tool {tool} is not a valid Tool object")

    def _build_v1_request(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build the ``chat.completions.create`` kwargs for a v1 (str-input) call."""
        if isinstance(message_history, MessageHistory):
            message_history = message_history.messages
        return {
            "messages": self.get_messages(input, message_history, system_instruction),
            "model": self.model_name,
            **self.model_params,
        }

    def _build_v2_request(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the ``chat.completions.create`` kwargs for a v2 (message-list) call."""
        kwargs = dict(kwargs)
        messages = self.get_messages_v2(input)
        params = self.model_params.copy() if self.model_params else {}

        if params.pop("response_format", None) is not None and response_format is None:
            logger.warning(
                "response_format in model_params is ignored. "
                "Pass response_format to invoke() instead."
            )

        if response_format is not None:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                # beta.parse() has strict limitations, so convert to JSON schema instead
                schema = response_format.model_json_schema()
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                }
            else:
                kwargs["response_format"] = response_format

        return {
            "messages": messages,
            "model": self.model_name,
            **params,
            **kwargs,
        }

    def _parse_response(self, response: Any) -> LLMResponse:
        """Build an ``LLMResponse`` from a Chat Completions response."""
        content = response.choices[0].message.content or ""
        usage = None
        if response.usage:
            usage = LLMUsage(
                request_tokens=response.usage.prompt_tokens,
                response_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        return LLMResponse(content=content, usage=usage)

    def _call_sync(self, request: dict[str, Any]) -> LLMResponse:
        """Sync transport hook: the only place ``client.chat.completions.create``
        is called."""
        try:
            response = self.client.chat.completions.create(**request)
            return self._parse_response(response)
        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)

    async def _call_async(self, request: dict[str, Any]) -> LLMResponse:
        """Async transport hook — see :meth:`_call_sync`."""
        try:
            response = await self.async_client.chat.completions.create(**request)
            return self._parse_response(response)
        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)

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

    @rate_limit_handler_decorator
    def __invoke_v1_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],  # Tools definition as a sequence of Tool objects
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Sends a text input to the OpenAI chat completion model with tool definitions
        and retrieves a tool call response.

        Args:
            input (str): Text sent to the LLM.
            tools (List[Tool]): List of Tools for the LLM to choose from.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            ToolCallResponse: The response from the LLM containing a tool call.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages

            params = self.model_params.copy() if self.model_params else {}
            if "temperature" not in params:
                params["temperature"] = 0.0

            # Convert tools to OpenAI's expected type
            openai_tools: List[ChatCompletionToolParam] = []
            for tool in tools:
                openai_format_tool = self._convert_tool_to_openai_format(tool)
                openai_tools.append(cast(ChatCompletionToolParam, openai_format_tool))

            response = self.client.chat.completions.create(
                messages=self.get_messages(input, message_history, system_instruction),
                model=self.model_name,
                tools=openai_tools,
                tool_choice="auto",
                **params,
            )

            message = response.choices[0].message

            # If there's no tool call, return the content as a regular response
            if not message.tool_calls or len(message.tool_calls) == 0:
                return ToolCallResponse(
                    tool_calls=[],
                    content=message.content,
                )

            # Process all tool calls
            tool_calls = []

            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, AttributeError) as e:
                    raise LLMGenerationError(
                        f"Failed to parse tool call arguments: {e}"
                    )

                tool_calls.append(
                    ToolCall(name=tool_call.function.name, arguments=args)
                )

            return ToolCallResponse(tool_calls=tool_calls, content=message.content)

        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)

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

    @async_rate_limit_handler_decorator
    async def __ainvoke_v1_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],  # Tools definition as a sequence of Tool objects
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Asynchronously sends a text input to the OpenAI chat completion model with tool definitions
        and retrieves a tool call response.

        Args:
            input (str): Text sent to the LLM.
            tools (List[Tool]): List of Tools for the LLM to choose from.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            ToolCallResponse: The response from the LLM containing a tool call.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages

            params = self.model_params.copy()
            if "temperature" not in params:
                params["temperature"] = 0.0

            # Convert tools to OpenAI's expected type
            openai_tools: List[ChatCompletionToolParam] = []
            for tool in tools:
                openai_format_tool = self._convert_tool_to_openai_format(tool)
                openai_tools.append(cast(ChatCompletionToolParam, openai_format_tool))

            response = await self.async_client.chat.completions.create(
                messages=self.get_messages(input, message_history, system_instruction),
                model=self.model_name,
                tools=openai_tools,
                tool_choice="auto",
                **params,
            )

            message = response.choices[0].message

            # If there's no tool call, return the content as a regular response
            if not message.tool_calls or len(message.tool_calls) == 0:
                return ToolCallResponse(
                    tool_calls=[ToolCall(name="", arguments={})],
                    content=message.content or "",
                )

            # Process all tool calls
            tool_calls = []

            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, AttributeError) as e:
                    raise LLMGenerationError(
                        f"Failed to parse tool call arguments: {e}"
                    )

                tool_calls.append(
                    ToolCall(name=tool_call.function.name, arguments=args)
                )

            return ToolCallResponse(tool_calls=tool_calls, content=message.content)

        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)


class OpenAILLM(BaseOpenAILLM):
    """OpenAI LLM."""

    supports_structured_output: bool = True

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """OpenAI LLM

        Wrapper for the openai Python client LLM.

        Args:
            model_name (str):
            model_params (str): Parameters for LLMInterface(V1) like temperature that will be passed to the model when text is sent to it. Defaults to None.
            rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting for LLMInterface(V1). Defaults to retry with exponential backoff.
            base_url (Optional[str], optional): Base URL to use instead of OpenAI's default API
                endpoint, e.g. to reach an OpenAI-compatible server. Passed through to both the
                sync and async SDK clients. Can be combined with an ``http_client`` passed via
                kwargs (``base_url`` sets where requests go, ``http_client`` how they are
                sent); a base URL configured on the httpx client itself is ignored by the
                SDK — use this parameter instead. Defaults to None.
            kwargs: All other parameters will be passed to the openai.OpenAI init.
        """
        super().__init__(
            model_name=model_name,
            model_params=model_params,
            rate_limit_handler=rate_limit_handler,
        )
        sync_params, async_params = split_http_client_kwargs(kwargs)
        if base_url is not None:
            sync_params["base_url"] = base_url
            async_params["base_url"] = base_url
        self.client = self.openai.OpenAI(**sync_params)
        self.async_client = self.openai.AsyncOpenAI(**async_params)


class AzureOpenAILLM(BaseOpenAILLM):
    """Azure OpenAI LLM."""

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ):
        """Azure OpenAI LLM. Use this class when using an OpenAI model
        hosted on Microsoft Azure.

        Args:
            model_name (str):
            model_params (str): Parameters for LLMInterface(V1) like temperature that will be passed to the model when text is sent to it. Defaults to None.
            rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting for LLMInterface(V1). Defaults to retry with exponential backoff.
            kwargs: All other parameters will be passed to the openai.OpenAI init.
        """
        super().__init__(
            model_name=model_name,
            model_params=model_params,
            rate_limit_handler=rate_limit_handler,
        )
        sync_params, async_params = split_http_client_kwargs(kwargs)
        self.client = self.openai.AzureOpenAI(**sync_params)
        self.async_client = self.openai.AsyncAzureOpenAI(**async_params)
