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
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from, no-else-return, import-outside-toplevel
class GeminiLLM(LLMInterface, LLMInterfaceV2):
    """LLM interface for Google Gemini via the google.genai SDK.

    Args:
        model_name (str): Model name. Defaults to "gemini-2.0-flash".
        model_params (Optional[dict]): Additional parameters passed to the model.
        rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting.
        **kwargs (Any): Arguments passed to the genai.Client.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ) -> None:
        if genai is None or types is None:
            raise ImportError(
                "Could not import google-genai python client. "
                'Please install it with `pip install "neo4j-graphrag[google-genai]"`.'
            )
        LLMInterfaceV2.__init__(
            self,
            model_name=model_name,
            model_params=model_params or {},
            rate_limit_handler=rate_limit_handler,
            **kwargs,
        )
        self.client = genai.Client(**kwargs)

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

    @rate_limit_handler_decorator
    def __invoke_v1(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        try:
            contents = self.get_messages(input, message_history)
            config = self._build_config(system_instruction=system_instruction)
            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )
            return LLMResponse(content=response.text)
        except Exception as e:
            raise LLMGenerationError(f"Error calling GeminiLLM: {e}") from e

    @async_rate_limit_handler_decorator
    async def __ainvoke_v1(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        try:
            contents = self.get_messages(input, message_history)
            config = self._build_config(system_instruction=system_instruction)
            response = await self.client.aio.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )
            return LLMResponse(content=response.text)
        except Exception as e:
            raise LLMGenerationError(f"Error calling GeminiLLM: {e}") from e

    @rate_limit_handler_decorator
    def __invoke_v2(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            system_instruction, contents = self.get_messages_v2(input)
            config = self._build_config(
                system_instruction=system_instruction,
                response_format=response_format,
                **kwargs,
            )
            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )
            return LLMResponse(content=response.text)
        except Exception as e:
            raise LLMGenerationError(f"Error calling GeminiLLM: {e}") from e

    @async_rate_limit_handler_decorator
    async def __ainvoke_v2(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            system_instruction, contents = self.get_messages_v2(input)
            config = self._build_config(
                system_instruction=system_instruction,
                response_format=response_format,
                **kwargs,
            )
            response = await self.client.aio.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )
            return LLMResponse(content=response.text)
        except Exception as e:
            raise LLMGenerationError(f"Error calling GeminiLLM: {e}") from e

    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        try:
            contents = self.get_messages(input, message_history)
            config = self._build_config(
                system_instruction=system_instruction, tools=tools
            )
            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )
            return self._parse_tool_response(response)
        except Exception as e:
            raise LLMGenerationError(f"Error calling GeminiLLM with tools: {e}") from e

    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        try:
            contents = self.get_messages(input, message_history)
            config = self._build_config(
                system_instruction=system_instruction, tools=tools
            )
            response = await self.client.aio.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )
            return self._parse_tool_response(response)
        except Exception as e:
            raise LLMGenerationError(f"Error calling GeminiLLM with tools: {e}") from e

    def get_messages(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
    ) -> list[types.Content]:
        messages: list[types.Content] = []
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
                if role == "user":
                    messages.append(
                        types.Content(
                            role="user", parts=[types.Part.from_text(text=content)]
                        )
                    )
                elif role == "assistant":
                    messages.append(
                        types.Content(
                            role="model", parts=[types.Part.from_text(text=content)]
                        )
                    )

        messages.append(
            types.Content(role="user", parts=[types.Part.from_text(text=input)])
        )
        return messages

    def get_messages_v2(
        self,
        input: List[LLMMessage],
    ) -> tuple[str | None, list[types.Content]]:
        messages: list[types.Content] = []
        system_instruction = None
        for message in input:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                messages.append(
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=content)]
                    )
                )
            elif role == "assistant":
                messages.append(
                    types.Content(
                        role="model", parts=[types.Part.from_text(text=content)]
                    )
                )
        return system_instruction, messages

    def _build_config(
        self,
        system_instruction: Optional[str] = None,
        tools: Optional[Sequence[Tool]] = None,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **extra: Any,
    ) -> types.GenerateContentConfig:
        config_kwargs: dict[str, Any] = {}

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if tools:
            config_kwargs["tools"] = self._get_llm_tools(tools)
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY
                )
            )

        if response_format is not None:
            config_kwargs["response_mime_type"] = "application/json"
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                config_kwargs["response_schema"] = response_format
            else:
                config_kwargs["response_schema"] = response_format

        config_kwargs.update(self.model_params)
        config_kwargs.update(extra)
        return types.GenerateContentConfig(**config_kwargs)

    def _get_llm_tools(
        self, tools: Optional[Sequence[Tool]]
    ) -> Optional[list[types.Tool]]:
        if not tools:
            return None
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool.get_name(),
                        description=tool.get_description(),
                        parameters=tool.get_parameters(
                            exclude=["additional_properties"]
                        ),
                    )
                    for tool in tools
                ]
            )
        ]

    def _parse_tool_response(
        self, response: types.GenerateContentResponse
    ) -> ToolCallResponse:
        tool_calls = []
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    tool_calls.append(
                        ToolCall(
                            name=part.function_call.name or "",
                            arguments=dict(part.function_call.args or {}),
                        )
                    )
        return ToolCallResponse(tool_calls=tool_calls, content=None)
