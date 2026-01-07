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

from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Union, cast, overload

from pydantic import ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMInterface, LLMInterfaceV2
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
    rate_limit_handler as rate_limit_handler_decorator,
    async_rate_limit_handler as async_rate_limit_handler_decorator,
)
from neo4j_graphrag.llm.types import (
    BaseMessage,
    LLMResponse,
    MessageList,
    UserMessage,
)
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage

if TYPE_CHECKING:
    from anthropic.types.message_param import MessageParam
    from anthropic import NotGiven


# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from, no-else-return, import-outside-toplevel
class AnthropicLLM(LLMInterface, LLMInterfaceV2):  # type: ignore[misc]
    """Interface for large language models on Anthropic

    Args:
        model_name (str, optional): Name of the LLM to use. Defaults to "gemini-1.5-flash-001".
        model_params (Optional[dict], optional): Additional parameters for LLMInterface(V1) passed to the model when text is sent to it. Defaults to None.
        system_instruction: Optional[str], optional): Additional instructions for setting the behavior and context for the model in a conversation. Defaults to None.
        rate_limit_handler (Optional[RateLimitHandler], optional): Handler for managing rate limits for LLMInterface(V1). Defaults to None.
        model_kwargs (Optional[dict], optional): Additional parameters for LLMInterfaceV2 passed to the model when text is sent to it. Defaults to None.
        rate_limiter (Optional[RateLimitHandler], optional): Handler for managing rate limits for LLMInterfaceV2. Defaults to None.
        **kwargs (Any): Arguments passed to the model when for the class is initialised. Defaults to None.

    Raises:
        LLMGenerationError: If there's an error generating the response from the model.

    Example:

    .. code-block:: python

        from neo4j_graphrag.llm import AnthropicLLM

        llm = AnthropicLLM(
            model_name="claude-3-opus-20240229",
            model_params={"max_tokens": 1000},
            api_key="sk...",   # can also be read from env vars
        )
        llm.invoke("Who is the mother of Paul Atreides?")
    """

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        rate_limiter: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                """Could not import Anthropic Python client.
                Please install it with `pip install "neo4j-graphrag[anthropic]"`."""
            )
        LLMInterfaceV2.__init__(
            self,
            model_name=model_name,
            model_kwargs=model_kwargs or model_params or {},
            rate_limiter=rate_limiter or rate_limit_handler,
            **kwargs,
        )
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(**kwargs)
        self.async_client = anthropic.AsyncAnthropic(**kwargs)

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
            return self.__invoke_v1(input, message_history, system_instruction)
        elif isinstance(input, list):
            return self.__invoke_v2(input, **kwargs)
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
            return await self.__ainvoke_v1(input, message_history, system_instruction)
        elif isinstance(input, list):
            return await self.__ainvoke_v2(input, **kwargs)
        else:
            raise ValueError(f"Invalid input type for ainvoke method - {type(input)}")

    # implementaions
    @rate_limit_handler_decorator
    def __invoke_v1(
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
            messages = self.get_messages(input, message_history)
            response = self.client.messages.create(
                model=self.model_name,
                system=system_instruction or self.anthropic.NOT_GIVEN,
                messages=messages,
                **self.model_params,
            )
            response_content = response.content
            if response_content and len(response_content) > 0:
                text = response_content[0].text
            else:
                raise LLMGenerationError("LLM returned empty response.")
            return LLMResponse(content=text)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)

    @rate_limit_handler_decorator
    def __invoke_v2(
        self,
        input: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            system_instruction, messages = self.get_brand_new_messages(input)
            response = self.client.messages.create(
                model=self.model_name,
                system=system_instruction,
                messages=messages,
                **self.model_params,
                **kwargs,
            )
            response_content = response.content
            if response_content and len(response_content) > 0:
                text = response_content[0].text
            else:
                raise LLMGenerationError("LLM returned empty response.")
            return LLMResponse(content=text)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)

    @async_rate_limit_handler_decorator
    async def __ainvoke_v1(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Asynchronously sends text to the LLM and returns a response.

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
            messages = self.get_messages(input, message_history)
            response = await self.async_client.messages.create(
                model=self.model_name,
                system=system_instruction or self.anthropic.NOT_GIVEN,
                messages=messages,
                **self.model_params,
            )
            response_content = response.content
            if response_content and len(response_content) > 0:
                text = response_content[0].text
            else:
                raise LLMGenerationError("LLM returned empty response.")
            return LLMResponse(content=text)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)

    @async_rate_limit_handler_decorator
    async def __ainvoke_v2(self, input: List[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Asynchronously sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            system_instruction, messages = self.get_brand_new_messages(input)
            response = await self.async_client.messages.create(
                model=self.model_name,
                system=system_instruction,
                messages=messages,
                **self.model_params,
                **kwargs,
            )
            response_content = response.content
            if response_content and len(response_content) > 0:
                text = response_content[0].text
            else:
                raise LLMGenerationError("LLM returned empty response.")
            return LLMResponse(content=text)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)

    # subsidiary methods
    def get_messages(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
    ) -> Iterable[MessageParam]:
        """Constructs the message list for the LLM from the input and message history."""
        messages: list[dict[str, str]] = []
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
    ) -> tuple[Union[str, NotGiven], Iterable[MessageParam]]:
        """Constructs the message list for the LLM from the input."""
        messages: list[MessageParam] = []
        system_instruction: Union[str, NotGiven] = self.anthropic.NOT_GIVEN
        for i in input:
            if i["role"] == "system":
                system_instruction = i["content"]
            else:
                if i["role"] not in ("user", "assistant"):
                    raise ValueError(f"Unknown role: {i['role']}")
                messages.append(
                    self.anthropic.types.MessageParam(
                        role=i["role"],
                        content=i["content"],
                    )
                )
        return system_instruction, messages
