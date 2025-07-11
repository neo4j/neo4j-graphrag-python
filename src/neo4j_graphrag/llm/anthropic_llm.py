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

from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Union, cast

from pydantic import ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.rate_limit import (
    RateLimitHandler,
    rate_limit_handler,
    async_rate_limit_handler,
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


class AnthropicLLM(LLMInterface):
    """Interface for large language models on Anthropic

    Args:
        model_name (str, optional): Name of the LLM to use. Defaults to "gemini-1.5-flash-001".
        model_params (Optional[dict], optional): Additional parameters passed to the model when text is sent to it. Defaults to None.
        system_instruction: Optional[str], optional): Additional instructions for setting the behavior and context for the model in a conversation. Defaults to None.
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
        **kwargs: Any,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                """Could not import Anthropic Python client.
                Please install it with `pip install "neo4j-graphrag[anthropic]"`."""
            )
        super().__init__(model_name, model_params, rate_limit_handler)
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(**kwargs)
        self.async_client = anthropic.AsyncAnthropic(**kwargs)

    def get_messages(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
    ) -> Iterable[MessageParam]:
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

    @rate_limit_handler
    def invoke(
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

    @async_rate_limit_handler
    async def ainvoke(
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
