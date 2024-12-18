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

from typing import TYPE_CHECKING, Any, Optional
from pydantic import ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import (
    LLMResponse,
    MessageList,
    SystemMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from cohere import ChatMessages


class CohereLLM(LLMInterface):
    """Interface for large language models on the Cohere platform

    Args:
        model_name (str, optional): Name of the LLM to use. Defaults to "gemini-1.5-flash-001".
        model_params (Optional[dict], optional): Additional parameters passed to the model when text is sent to it. Defaults to None.
        system_instruction: Optional[str], optional): Additional instructions for setting the behavior and context for the model in a conversation. Defaults to None.
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
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        try:
            import cohere
        except ImportError:
            raise ImportError(
                """Could not import cohere python client.
                Please install it with `pip install "neo4j-graphrag[cohere]"`."""
            )
        super().__init__(model_name, model_params, system_instruction)
        self.cohere = cohere
        self.cohere_api_error = cohere.core.api_error.ApiError

        self.client = cohere.ClientV2(**kwargs)
        self.async_client = cohere.AsyncClientV2(**kwargs)

    def get_messages(
        self,
        input: str,
        message_history: Optional[list[dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
    ) -> ChatMessages:
        messages = []
        system_message = (
            system_instruction
            if system_instruction is not None
            else self.system_instruction
        )
        if system_message:
            messages.append(SystemMessage(content=system_message).model_dump())
        if message_history:
            try:
                MessageList(messages=message_history)  # type: ignore
            except ValidationError as e:
                raise LLMGenerationError(e.errors()) from e
            messages.extend(message_history)
        messages.append(UserMessage(content=input).model_dump())
        return messages  # type: ignore

    def invoke(
        self,
        input: str,
        message_history: Optional[list[dict[str, str]]] = None,
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
            messages = self.get_messages(input, message_history, system_instruction)
            res = self.client.chat(
                messages=messages,
                model=self.model_name,
            )
        except self.cohere_api_error as e:
            raise LLMGenerationError(e)
        return LLMResponse(
            content=res.message.content[0].text,
        )

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[list[dict[str, str]]] = None,
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
            messages = self.get_messages(input, message_history, system_instruction)
            res = self.async_client.chat(
                messages=messages,
                model=self.model_name,
            )
        except self.cohere_api_error as e:
            raise LLMGenerationError(e)
        return LLMResponse(
            content=res.message.content[0].text,
        )
