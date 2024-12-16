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

import abc
from typing import TYPE_CHECKING, Any, Iterable, Optional

from pydantic import ValidationError

from ..exceptions import LLMGenerationError
from .base import LLMInterface
from .types import LLMResponse, SystemMessage, UserMessage, MessageList

if TYPE_CHECKING:
    import openai
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )


class BaseOpenAILLM(LLMInterface, abc.ABC):
    client: openai.OpenAI
    async_client: openai.AsyncOpenAI

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Base class for OpenAI LLM.

        Makes sure the openai Python client is installed during init.

        Args:
            model_name (str):
            model_params (str): Parameters like temperature that will be passed to the model when text is sent to it. Defaults to None.
            system_instruction: Optional[str], optional): Additional instructions for setting the behavior and context for the model in a conversation. Defaults to None.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                """Could not import openai Python client.
                Please install it with `pip install "neo4j-graphrag[openai]"`."""
            )
        self.openai = openai
        super().__init__(model_name, model_params, system_instruction)

    def get_messages(
        self, input: str, chat_history: Optional[list[Any]] = None, system_instruction: Optional[str] = None
    ) -> Iterable[ChatCompletionMessageParam]:
        messages = []
        system_message = system_instruction if system_instruction is not None else self.system_instruction
        if system_message:
            messages.append(SystemMessage(content=system_message).model_dump())
        if chat_history:
            try:
                MessageList(messages=chat_history)
            except ValidationError as e:
                raise LLMGenerationError(e.errors()) from e
            messages.extend(chat_history)
        messages.append(UserMessage(content=input).model_dump())
        return messages

    def invoke(
        self, input: str, chat_history: Optional[list[Any]] = None, system_instruction: Optional[str] = None
    ) -> LLMResponse:
        """Sends a text input to the OpenAI chat completion model
        and returns the response's content.

        Args:
            input (str): Text sent to the LLM.
            chat_history (Optional[list]): A collection previous messages, with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invokation.

        Returns:
            LLMResponse: The response from OpenAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            response = self.client.chat.completions.create(
                messages=self.get_messages(input, chat_history, system_instruction),
                model=self.model_name,
                **self.model_params,
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content)
        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)

    async def ainvoke(
        self, input: str, chat_history: Optional[list[Any]] = None
    ) -> LLMResponse:
        """Asynchronously sends a text input to the OpenAI chat
        completion model and returns the response's content.

        Args:
            input (str): Text sent to the LLM.
            chat_history (Optional[list]): A collection previous messages, with each message having a specific role assigned.

        Returns:
            LLMResponse: The response from OpenAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            response = await self.async_client.chat.completions.create(
                messages=self.get_messages(input, chat_history),
                model=self.model_name,
                **self.model_params,
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content)
        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)


class OpenAILLM(BaseOpenAILLM):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        """OpenAI LLM

        Wrapper for the openai Python client LLM.

        Args:
            model_name (str):
            model_params (str): Parameters like temperature that will be passed to the model when text is sent to it. Defaults to None.
            system_instruction: Optional[str], optional): Additional instructions for setting the behavior and context for the model in a conversation. Defaults to None.
            kwargs: All other parameters will be passed to the openai.OpenAI init.
        """
        super().__init__(model_name, model_params, system_instruction)
        self.client = self.openai.OpenAI(**kwargs)
        self.async_client = self.openai.AsyncOpenAI(**kwargs)


class AzureOpenAILLM(BaseOpenAILLM):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        """Azure OpenAI LLM. Use this class when using an OpenAI model
        hosted on Microsoft Azure.

        Args:
            model_name (str):
            model_params (str): Parameters like temperature that will be passed to the model when text is sent to it. Defaults to None.
            system_instruction: Optional[str], optional): Additional instructions for setting the behavior and context for the model in a conversation. Defaults to None.
            kwargs: All other parameters will be passed to the openai.OpenAI init.
        """
        super().__init__(model_name, model_params, system_instruction)
        self.client = self.openai.AzureOpenAI(**kwargs)
        self.async_client = self.openai.AsyncAzureOpenAI(**kwargs)
