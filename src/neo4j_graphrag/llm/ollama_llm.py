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
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
    cast,
    overload,
)

# 3rd-party dependencies
from pydantic import ValidationError

# project dependencies
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
    rate_limit_handler,
    async_rate_limit_handler,
)

from .base import LLMInterface, LLMInterfaceV2
from .types import (
    BaseMessage,
    LLMResponse,
    MessageList,
    SystemMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from ollama import Message

# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from, no-else-return


class OllamaLLM(LLMInterface, LLMInterfaceV2):  # type: ignore[misc]
    """LLM wrapper for Ollama models."""

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ):
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "Could not import ollama Python client. "
                "Please install it with `pip install ollama`."
            )
        super().__init__(model_name, model_params, rate_limit_handler)
        self.ollama = ollama
        self.client = ollama.Client(
            **kwargs,
        )
        self.async_client = ollama.AsyncClient(
            **kwargs,
        )
        if "stream" in self.model_params:
            raise ValueError("Streaming is not supported by the OllamaLLM wrapper")
        # bug-fix with backward compatibility:
        # we mistakenly passed all "model_params" under the options argument
        # next two lines to be removed in 2.0
        if not any(
            key in self.model_params for key in ("options", "format", "keep_alive")
        ):
            warnings.warn(
                """Passing options directly without including them in an 'options' key is deprecated. Ie you must use model_params={"options": {"temperature": 0}}""",
                DeprecationWarning,
            )
            self.model_params = {"options": self.model_params}

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
        **kwargs: Any,
    ) -> LLMResponse: ...

    # switching logics to LLMInterface or LLMInterfaceV2
    def invoke(  # type: ignore[no-redef]
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return self.__legacy_invoke(input, message_history, system_instruction)
        elif isinstance(input, list):
            return self.__brand_new_invoke(input, **kwargs)
        else:
            raise ValueError(f"Invalid input type for invoke method - {type(input)}")

    async def ainvoke(  # type: ignore[no-redef]
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return await self.__legacy_ainvoke(
                input, message_history, system_instruction
            )
        elif isinstance(input, list):
            return await self.__brand_new_ainvoke(input, **kwargs)
        else:
            raise ValueError(f"Invalid input type for ainvoke method - {type(input)}")

    @rate_limit_handler
    def __legacy_invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            response = self.client.chat(
                model=self.model_name,
                messages=self.get_messages(input, message_history, system_instruction),
                **self.model_params,
            )
            content = response.message.content or ""
            return LLMResponse(content=content)
        except self.ollama.ResponseError as e:
            raise LLMGenerationError(e)

    def __brand_new_invoke(
        self,
        input: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """Sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=self.get_brand_new_messages(input),
                **self.model_params,
                **kwargs,
            )
            content = response.message.content or ""
            return LLMResponse(content=content)
        except self.ollama.ResponseError as e:
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
            response = await self.async_client.chat(
                model=self.model_name,
                messages=self.get_messages(input, message_history, system_instruction),
                options=self.model_params,
            )
            content = response.message.content or ""
            return LLMResponse(content=content)
        except self.ollama.ResponseError as e:
            raise LLMGenerationError(e)

    async def __brand_new_ainvoke(
        self,
        input: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """Asynchronously sends a text input to the OpenAI chat
        completion model and returns the response's content.

        Args:
            input (str): Text sent to the LLM.

        Returns:
            LLMResponse: The response from OpenAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            params = {**self.model_params, **kwargs}
            response = await self.async_client.chat(
                model=self.model_name,
                messages=self.get_brand_new_messages(input),
                options=params,
            )
            content = response.message.content or ""
            return LLMResponse(content=content)
        except self.ollama.ResponseError as e:
            raise LLMGenerationError(e)

    # subsdiary methods
    def get_messages(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> Sequence[Message]:
        """Constructs the message list for the Ollama chat API."""
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
        input: list[LLMMessage],
    ) -> Sequence[Message]:
        """Constructs the message list for the Ollama chat API."""
        return [self.ollama.Message(**i) for i in input]
