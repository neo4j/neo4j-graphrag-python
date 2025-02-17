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

from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, cast

from pydantic import ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError

from .base import LLMInterface
from .types import (
    BaseMessage,
    LLMMessage,
    LLMResponse,
    MessageList,
    SystemMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from ollama import Message


class OllamaLLM(LLMInterface):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "Could not import ollama Python client. "
                "Please install it with `pip install ollama`."
            )
        super().__init__(model_name, model_params, **kwargs)
        self.ollama = ollama
        self.client = ollama.Client(
            **kwargs,
        )
        self.async_client = ollama.AsyncClient(
            **kwargs,
        )

    def get_messages(
        self,
        input: str,
        message_history: Optional[list[LLMMessage]] = None,
        system_instruction: Optional[str] = None,
    ) -> Sequence[Message]:
        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction).model_dump())
        if message_history:
            try:
                MessageList(messages=cast(list[BaseMessage], message_history))
            except ValidationError as e:
                raise LLMGenerationError(e.errors()) from e
            messages.extend(cast(Iterable[dict[str, Any]], message_history))
        messages.append(UserMessage(content=input).model_dump())
        return messages  # type: ignore

    def invoke(
        self,
        input: str,
        message_history: Optional[list[LLMMessage]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.
            message_history (Optional[list]): A collection previous messages, with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=self.get_messages(input, message_history, system_instruction),
                options=self.model_params,
            )
            content = response.message.content or ""
            return LLMResponse(content=content)
        except self.ollama.ResponseError as e:
            raise LLMGenerationError(e)

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[list[LLMMessage]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Asynchronously sends a text input to the OpenAI chat
        completion model and returns the response's content.

        Args:
            input (str): Text sent to the LLM.
            message_history (Optional[list]): A collection previous messages, with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            LLMResponse: The response from OpenAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            response = await self.async_client.chat(
                model=self.model_name,
                messages=self.get_messages(input, message_history, system_instruction),
                options=self.model_params,
            )
            content = response.message.content or ""
            return LLMResponse(content=content)
        except self.ollama.ResponseError as e:
            raise LLMGenerationError(e)
