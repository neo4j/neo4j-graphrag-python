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
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Iterable,
    Sequence,
    Union,
    cast,
    overload,
    Type,
)

# 3rd party dependencies
from pydantic import ValidationError

# project dependencies
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
    rate_limit_handler,
    async_rate_limit_handler,
)
from neo4j_graphrag.tool import Tool

from ..exceptions import LLMGenerationError
from .base import LLMInterface, LLMInterfaceV2
from .types import (
    BaseMessage,
    LLMResponse,
    MessageList,
    ToolCall,
    ToolCallResponse,
    SystemMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionToolParam,
    )
    from openai import OpenAI, AsyncOpenAI
else:
    ChatCompletionMessageParam = Any
    ChatCompletionToolParam = Any
    OpenAI = Any
    AsyncOpenAI = Any


# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from, no-else-return
class BaseOpenAILLM(LLMInterface, LLMInterfaceV2, abc.ABC):
    """Base class for OpenAI LLMs."""

    client: OpenAI
    async_client: AsyncOpenAI

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
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
        super().__init__(model_name, model_params, rate_limit_handler)

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
    ) -> LLMResponse: ...

    @overload  # type: ignore[no-overload-impl]
    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse: ...

    @overload
    def invoke_with_tools(
        self,
        input: list[LLMMessage],
        tools: Sequence[Tool],  # Tools definition as a sequence of Tool objects
    ) -> ToolCallResponse: ...

    @overload  # type: ignore[no-overload-impl]
    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse: ...

    @overload
    async def ainvoke_with_tools(
        self,
        input: list[LLMMessage],
        tools: Sequence[Tool],
    ) -> ToolCallResponse: ...

    # switching logics to LLMInterface or LLMInterfaceV2
    def invoke(  # type: ignore[no-redef]
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        if isinstance(input, str):
            return self.__legacy_invoke(input, message_history, system_instruction)
        elif isinstance(input, list):
            return self.__brand_new_invoke(input)
        else:
            raise ValueError(f"Invalid input type for invoke method - {type(input)}")

    async def ainvoke(  # type: ignore[no-redef]
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        if isinstance(input, str):
            return await self.__legacy_ainvoke(
                input, message_history, system_instruction
            )
        elif isinstance(input, list):
            return await self.__brand_new_ainvoke(input)
        else:
            raise ValueError(f"Invalid input type for ainvoke method - {type(input)}")

    def invoke_with_tools(  # type: ignore[no-redef]
        self,
        input: Union[str, List[LLMMessage]],
        tools: Sequence[Tool],  # Tools definition as a sequence of Tool objects
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        if isinstance(input, str):
            return self.__legacy_invoke_with_tools(
                input, tools, message_history, system_instruction
            )
        elif isinstance(input, list):
            return self.__brand_new_invoke_with_tools(input, tools)
        else:
            raise ValueError(
                f"Invalid input type for invoke_with_tools method - {type(input)}"
            )

    async def ainvoke_with_tools(  # type: ignore[no-redef]
        self,
        input: Union[str, List[LLMMessage]],
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        if isinstance(input, str):
            return await self.__legacy_ainvoke_with_tools(
                input, tools, message_history, system_instruction
            )
        elif isinstance(input, list):
            return await self.__brand_new_ainvoke_with_tools(input, tools)
        else:
            raise ValueError(
                f"Invalid input type for ainvoke_with_tools method - {type(input)}"
            )

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

    def get_brand_new_messages(
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

    def __brand_new_invoke(
        self,
        input: List[LLMMessage],
    ) -> LLMResponse:
        """New invoke method for LLMInterfaceV2.

        Args:
            input (List[LLMMessage]): Input to the LLM.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            response = self.client.chat.completions.create(
                messages=self.get_brand_new_messages(input),
                model=self.model_name,
                **self.model_params,
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content)
        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)

    @rate_limit_handler
    def __legacy_invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Sends a text input to the OpenAI chat completion model
        and returns the response's content.

        Args:
            input (str): Text sent to the LLM.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            LLMResponse: The response from OpenAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            response = self.client.chat.completions.create(
                messages=self.get_messages(input, message_history, system_instruction),
                model=self.model_name,
                **self.model_params,
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content)
        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)

    @rate_limit_handler
    def __legacy_invoke_with_tools(
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

    def __brand_new_invoke_with_tools(
        self,
        input: List[LLMMessage],
        tools: Sequence[Tool],  # Tools definition as a sequence of Tool objects
    ) -> ToolCallResponse:
        """Sends a text input to the OpenAI chat completion model with tool definitions
        and retrieves a tool call response.

        Args:
            input (str): Text sent to the LLM.
            tools (List[Tool]): List of Tools for the LLM to choose from.

        Returns:
            ToolCallResponse: The response from the LLM containing a tool call.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            params = self.model_params.copy() if self.model_params else {}
            if "temperature" not in params:
                params["temperature"] = 0.0

            # Convert tools to OpenAI's expected type
            openai_tools: List[ChatCompletionToolParam] = []
            for tool in tools:
                openai_format_tool = self._convert_tool_to_openai_format(tool)
                openai_tools.append(cast(ChatCompletionToolParam, openai_format_tool))

            response = self.client.chat.completions.create(
                messages=self.get_brand_new_messages(input),
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

    @async_rate_limit_handler
    async def __legacy_ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Asynchronously sends a text input to the OpenAI chat
        completion model and returns the response's content.

        Args:
            input (str): Text sent to the LLM.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            LLMResponse: The response from OpenAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            response = await self.async_client.chat.completions.create(
                messages=self.get_messages(input, message_history, system_instruction),
                model=self.model_name,
                **self.model_params,
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content)
        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)

    async def __brand_new_ainvoke(
        self,
        input: List[LLMMessage],
    ) -> LLMResponse:
        """Asynchronous new invoke method for LLMInterfaceV2."""
        try:
            response = await self.async_client.chat.completions.create(
                messages=self.get_brand_new_messages(input),
                model=self.model_name,
                **self.model_params,
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content)
        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)

    @async_rate_limit_handler
    async def __legacy_ainvoke_with_tools(
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
            import json

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

    async def __brand_new_ainvoke_with_tools(
        self,
        input: List[LLMMessage],
        tools: Sequence[Tool],  # Tools definition as a sequence of Tool objects
    ) -> ToolCallResponse:
        """Asynchronously sends a text input to the OpenAI chat completion model with tool definitions
        and retrieves a tool call response.

        Args:
            input (str): Text sent to the LLM.
            tools (List[Tool]): List of Tools for the LLM to choose from.

        Returns:
            ToolCallResponse: The response from the LLM containing a tool call.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            params = self.model_params.copy()
            if "temperature" not in params:
                params["temperature"] = 0.0

            # Convert tools to OpenAI's expected type
            openai_tools: List[ChatCompletionToolParam] = []
            for tool in tools:
                openai_format_tool = self._convert_tool_to_openai_format(tool)
                openai_tools.append(cast(ChatCompletionToolParam, openai_format_tool))

            response = await self.async_client.chat.completions.create(
                messages=self.get_brand_new_messages(input),
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
            import json

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

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ):
        """OpenAI LLM

        Wrapper for the openai Python client LLM.

        Args:
            model_name (str):
            model_params (str): Parameters like temperature that will be passed to the model when text is sent to it. Defaults to None.
            rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting. Defaults to retry with exponential backoff.
            kwargs: All other parameters will be passed to the openai.OpenAI init.
        """
        super().__init__(model_name, model_params, rate_limit_handler)
        self.client = self.openai.OpenAI(**kwargs)
        self.async_client = self.openai.AsyncOpenAI(**kwargs)


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
            model_params (str): Parameters like temperature that will be passed to the model when text is sent to it. Defaults to None.
            rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting. Defaults to retry with exponential backoff.
            kwargs: All other parameters will be passed to the openai.OpenAI init.
        """
        super().__init__(model_name, model_params, rate_limit_handler)
        self.client = self.openai.AzureOpenAI(**kwargs)
        self.async_client = self.openai.AsyncAzureOpenAI(**kwargs)
