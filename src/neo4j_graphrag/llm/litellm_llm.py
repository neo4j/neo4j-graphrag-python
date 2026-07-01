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

from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

from pydantic import BaseModel, ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMBase
from neo4j_graphrag.llm.types import (
    BaseMessage,
    LLMResponse,
    LLMUsage,
    MessageList,
    SystemMessage,
    ToolCall,
    ToolCallResponse,
    UserMessage,
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


# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from, no-else-return, import-outside-toplevel
class LiteLLMChat(LLMBase):
    """Interface for large language models via LiteLLM AI gateway.

    LiteLLM provides a unified interface to 100+ LLM providers (OpenAI, Anthropic,
    Google, Azure, AWS Bedrock, Ollama, and more) using the OpenAI completion format.

    Args:
        model_name (str): The model identifier in LiteLLM format
            (e.g. "gpt-4o", "anthropic/claude-sonnet-4-20250514", "ollama/llama3").
        model_params (Optional[dict], optional): Additional parameters passed to the
            model when text is sent to it. Defaults to None.
        rate_limit_handler (Optional[RateLimitHandler], optional): A rate limit handler
            to manage API rate limits. Defaults to None.
        **kwargs (Any): Arguments passed to ``litellm.completion`` as defaults
            (e.g. ``api_key``, ``api_base``, ``drop_params``).

    Raises:
        LLMGenerationError: If there's an error generating the response from the model.

    Example:

    .. code-block:: python

        from neo4j_graphrag.llm import LiteLLMChat

        llm = LiteLLMChat(model_name="gpt-4o", api_key="...")
        llm.invoke("Say something")

        # Use any LiteLLM-supported provider:
        llm = LiteLLMChat(model_name="anthropic/claude-sonnet-4-20250514", api_key="...")
        llm = LiteLLMChat(model_name="ollama/llama3")
    """

    supports_structured_output = False

    def __init__(
        self,
        model_name: str = "",
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ) -> None:
        try:
            import litellm
        except ImportError:
            raise ImportError(
                """Could not import litellm python client.
                Please install it with `pip install "neo4j-graphrag[litellm]"`."""
            )
        LLMBase.__init__(
            self,
            model_name=model_name,
            model_params=model_params or {},
            rate_limit_handler=rate_limit_handler,
        )
        self.litellm = litellm
        self.kwargs = kwargs

    def invoke(
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return self._invoke_v1(input, message_history, system_instruction)
        elif isinstance(input, list):
            return self._invoke_v2(input, response_format=response_format, **kwargs)
        else:
            raise ValueError(f"Invalid input type for invoke method - {type(input)}")

    async def ainvoke(
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return await self._ainvoke_v1(input, message_history, system_instruction)
        elif isinstance(input, list):
            return await self._ainvoke_v2(
                input, response_format=response_format, **kwargs
            )
        else:
            raise ValueError(f"Invalid input type for ainvoke method - {type(input)}")

    @rate_limit_handler_decorator
    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            messages = self.get_messages(input, message_history, system_instruction)
            tool_defs = [
                {
                    "type": "function",
                    "function": {
                        "name": t.get_name(),
                        "description": t.get_description(),
                        "parameters": t.get_parameters(),
                    },
                }
                for t in tools
            ]
            res = self.litellm.completion(
                model=self.model_name,
                messages=messages,
                tools=tool_defs,
                **self.kwargs,
                **self.model_params,
            )
        except Exception as e:
            raise LLMGenerationError(e)

        tool_calls = []
        if res.choices[0].message.tool_calls:
            import json

            for tc in res.choices[0].message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )
        return ToolCallResponse(
            tool_calls=tool_calls,
            content=res.choices[0].message.content,
        )

    @async_rate_limit_handler_decorator
    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            messages = self.get_messages(input, message_history, system_instruction)
            tool_defs = [
                {
                    "type": "function",
                    "function": {
                        "name": t.get_name(),
                        "description": t.get_description(),
                        "parameters": t.get_parameters(),
                    },
                }
                for t in tools
            ]
            res = await self.litellm.acompletion(
                model=self.model_name,
                messages=messages,
                tools=tool_defs,
                **self.kwargs,
                **self.model_params,
            )
        except Exception as e:
            raise LLMGenerationError(e)

        tool_calls = []
        if res.choices[0].message.tool_calls:
            import json

            for tc in res.choices[0].message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )
        return ToolCallResponse(
            tool_calls=tool_calls,
            content=res.choices[0].message.content,
        )

    # implementations
    @rate_limit_handler_decorator
    def _invoke_v1(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            messages = self.get_messages(input, message_history, system_instruction)
            res = self.litellm.completion(
                model=self.model_name,
                messages=messages,
                **self.kwargs,
                **self.model_params,
            )
        except Exception as e:
            raise LLMGenerationError(e)
        return self._to_llm_response(res)

    @rate_limit_handler_decorator
    def _invoke_v2(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            messages = self._to_litellm_messages(input)
            call_kwargs: dict[str, Any] = {
                **self.kwargs,
                **self.model_params,
            }
            if response_format is not None:
                if isinstance(response_format, type) and issubclass(
                    response_format, BaseModel
                ):
                    call_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_format.__name__,
                            "strict": True,
                            "schema": response_format.model_json_schema(),
                        },
                    }
                else:
                    call_kwargs["response_format"] = response_format
            res = self.litellm.completion(
                model=self.model_name,
                messages=messages,
                **call_kwargs,
            )
        except Exception as e:
            raise LLMGenerationError("Error calling LiteLLM") from e
        return self._to_llm_response(res)

    @async_rate_limit_handler_decorator
    async def _ainvoke_v1(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            messages = self.get_messages(input, message_history, system_instruction)
            res = await self.litellm.acompletion(
                model=self.model_name,
                messages=messages,
                **self.kwargs,
                **self.model_params,
            )
        except Exception as e:
            raise LLMGenerationError(e)
        return self._to_llm_response(res)

    @async_rate_limit_handler_decorator
    async def _ainvoke_v2(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            messages = self._to_litellm_messages(input)
            call_kwargs: dict[str, Any] = {
                **self.kwargs,
                **self.model_params,
            }
            if response_format is not None:
                if isinstance(response_format, type) and issubclass(
                    response_format, BaseModel
                ):
                    call_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_format.__name__,
                            "strict": True,
                            "schema": response_format.model_json_schema(),
                        },
                    }
                else:
                    call_kwargs["response_format"] = response_format
            res = await self.litellm.acompletion(
                model=self.model_name,
                messages=messages,
                **call_kwargs,
            )
        except Exception as e:
            raise LLMGenerationError("Error calling LiteLLM") from e
        return self._to_llm_response(res)

    # helper methods
    def get_messages(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction).model_dump())
        if message_history:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            try:
                MessageList(messages=cast(list[BaseMessage], message_history))
            except ValidationError as e:
                raise LLMGenerationError(e.errors()) from e
            messages.extend(cast(Iterable[dict[str, str]], message_history))
        messages.append(UserMessage(content=input).model_dump())
        return messages

    def _to_litellm_messages(
        self,
        input: list[LLMMessage],
    ) -> list[dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in input]

    def _to_llm_response(self, res: Any) -> LLMResponse:
        usage = None
        if hasattr(res, "usage") and res.usage is not None:
            input_tokens = getattr(res.usage, "prompt_tokens", None)
            output_tokens = getattr(res.usage, "completion_tokens", None)
            total = getattr(res.usage, "total_tokens", None)
            if input_tokens is not None:
                input_tokens = int(input_tokens)
            if output_tokens is not None:
                output_tokens = int(output_tokens)
            if total is not None:
                total = int(total)
            usage = LLMUsage(
                request_tokens=input_tokens,
                response_tokens=output_tokens,
                total_tokens=total,
            )
        content = ""
        if res.choices and res.choices[0].message.content:
            content = res.choices[0].message.content
        return LLMResponse(content=content, usage=usage)
