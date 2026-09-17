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
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
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
    SystemMessage,
    UserMessage,
)
from neo4j_graphrag.message_history import MessageHistory
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

if TYPE_CHECKING:
    from cohere import ChatMessages


# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from, no-else-return, import-outside-toplevel
class CohereLLM(LLMBase):
    """Interface for large language models on the Cohere platform

    Args:
        model_name (str, optional): Name of the LLM to use. Defaults to "".
        model_params (Optional[dict], optional): Additional parameters for LLMInterface(V1) passed to the model when text is sent to it. Defaults to None.
        system_instruction (Optional[str], optional): Additional instructions for setting the behavior and context for the model in a conversation. Defaults to None.
        rate_limit_handler (Optional[RateLimitHandler], optional): A rate limit handler for LLMInterface(V1) to manage API rate limits. Defaults to None.
        **kwargs (Any): Arguments passed to the model when for the class is initialised. Defaults to None.

    Raises:
        LLMGenerationError: If there's an error generating the response from the model.

    Example:

    .. code-block:: python

        from neo4j_graphrag.llm import CohereLLM

        llm = CohereLLM(api_key="...")
        llm.invoke("Say something")
    """

    def __init__(
        self,
        model_name: str = "",
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ) -> None:
        try:
            import cohere

            # Import the submodule rather than reaching for `cohere.core`: the
            # top-level package resolves attributes lazily and does not list
            # `core`, so the attribute chain raises AttributeError even though
            # the module is present and importable.
            from cohere.core.api_error import ApiError
        except ImportError:
            raise ImportError(
                """Could not import cohere python client.
                Please install it with `pip install "neo4j-graphrag[cohere]"`."""
            )
        LLMBase.__init__(
            self,
            model_name=model_name,
            model_params=model_params or {},
            rate_limit_handler=rate_limit_handler,
            **kwargs,
        )
        self.cohere = cohere
        self.cohere_api_error = ApiError

        self.client = cohere.ClientV2(**kwargs)
        self.async_client = cohere.AsyncClientV2(**kwargs)

    def _extract_text_content(self, content_items: Any) -> str:
        if not content_items:
            return ""
        text = getattr(content_items[0], "text", None)
        return text if isinstance(text, str) else ""

    def _build_v1_request(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build the ``client.chat`` kwargs for a v1 (str-input) call."""
        if isinstance(message_history, MessageHistory):
            message_history = message_history.messages
        return {
            "messages": self.get_messages(input, message_history, system_instruction),
            "model": self.model_name,
        }

    def _build_v2_request(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the ``client.chat`` kwargs for a v2 (message-list) call."""
        if response_format is not None:
            raise NotImplementedError(
                "CohereLLM does not currently support structured output"
            )
        return {
            "messages": self.get_messages_v2(input),
            "model": self.model_name,
        }

    def _parse_response(self, res: Any) -> LLMResponse:
        """Build an ``LLMResponse`` from a Cohere chat response."""
        usage = None
        if res.usage and res.usage.tokens:
            input_tokens = (
                int(res.usage.tokens.input_tokens)
                if res.usage.tokens.input_tokens is not None
                else None
            )
            output_tokens = (
                int(res.usage.tokens.output_tokens)
                if res.usage.tokens.output_tokens is not None
                else None
            )
            usage = LLMUsage(
                request_tokens=input_tokens,
                response_tokens=output_tokens,
                total_tokens=(input_tokens + output_tokens)
                if (input_tokens is not None and output_tokens is not None)
                else None,
            )
        return LLMResponse(
            content=self._extract_text_content(res.message.content), usage=usage
        )

    def _call_sync(self, request: dict[str, Any]) -> LLMResponse:
        """Sync transport hook: the only place ``client.chat`` is called."""
        try:
            res = self.client.chat(**request)
        except self.cohere_api_error as e:
            raise LLMGenerationError(e) from e
        return self._parse_response(res)

    async def _call_async(self, request: dict[str, Any]) -> LLMResponse:
        """Async transport hook — see :meth:`_call_sync`."""
        try:
            res = await self.async_client.chat(**request)
        except self.cohere_api_error as e:
            raise LLMGenerationError(e) from e
        return self._parse_response(res)

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

    # subsdiary methods
    def get_messages(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ChatMessages:
        """Converts input and message history to ChatMessages for Cohere."""
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
        input: list[LLMMessage],
    ) -> ChatMessages:
        """Converts a list of LLMMessage to ChatMessages for Cohere."""
        messages: ChatMessages = []
        for i in input:
            if i["role"] == "system":
                messages.append(self.cohere.SystemChatMessageV2(content=i["content"]))
            elif i["role"] == "user":
                messages.append(self.cohere.UserChatMessageV2(content=i["content"]))
            elif i["role"] == "assistant":
                messages.append(
                    self.cohere.AssistantChatMessageV2(content=i["content"])
                )
            else:
                raise ValueError(f"Unknown role: {i['role']}")
        return messages
