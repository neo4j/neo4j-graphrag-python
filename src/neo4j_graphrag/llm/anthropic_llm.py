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

from typing import Any, Iterable, Optional, TYPE_CHECKING

from pydantic import ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse, MessageList, UserMessage, BaseMessage

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
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                """Could not import Anthropic Python client.
                Please install it with `pip install "neo4j-graphrag[anthropic]"`."""
            )
        super().__init__(model_name, model_params, system_instruction)
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(**kwargs)
        self.async_client = anthropic.AsyncAnthropic(**kwargs)

    def get_messages(
        self, input: str, message_history: Optional[list[BaseMessage]] = None
    ) -> Iterable[MessageParam]:
        messages = []
        if message_history:
            try:
                MessageList(messages=message_history)
            except ValidationError as e:
                raise LLMGenerationError(e.errors()) from e
            messages.extend(message_history)
        messages.append(UserMessage(content=input).model_dump())
        return messages

    def invoke(
        self,
        input: str,
        message_history: Optional[list[BaseMessage]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.
            message_history (Optional[list]): A collection previous messages, with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invokation.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            messages = self.get_messages(input, message_history)
            system_message = (
                system_instruction
                if system_instruction is not None
                else self.system_instruction
            )
            response = self.client.messages.create(
                model=self.model_name,
                system=system_message,
                messages=messages,
                **self.model_params,
            )
            return LLMResponse(content=response.content)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[list[BaseMessage]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Asynchronously sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.
            message_history (Optional[list]): A collection previous messages, with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invokation.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            messages = self.get_messages(input, message_history)
            system_message = (
                system_instruction
                if system_instruction is not None
                else self.system_instruction
            )
            response = await self.async_client.messages.create(
                model=self.model_name,
                system=system_message,
                messages=messages,
                **self.model_params,
            )
            return LLMResponse(content=response.content)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)
