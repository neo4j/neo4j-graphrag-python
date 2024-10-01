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

from typing import Any, Optional, Type

from ..exceptions import LLMGenerationError
from .base import LLMInterface
from .types import LLMResponse

try:
    import openai
except ImportError:
    openai = None  # type: ignore


class OpenAILLM(LLMInterface):
    client_class: Type[openai.OpenAI] = openai.OpenAI
    async_client_class: Type[openai.AsyncOpenAI] = openai.AsyncOpenAI

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """

        Args:
            model_name (str):
            model_params (str): Parameters like temperature that will be passed to the model when text is sent to it
            kwargs: All other parameters will be passed to the openai.OpenAI init.

        """
        if openai is None:
            raise ImportError(
                "Could not import openai Python client. "
                "Please install it with `pip install openai`."
            )
        super().__init__(model_name, model_params, **kwargs)
        self.client = self.client_class(**kwargs)
        self.async_client = self.async_client_class(**kwargs)

    def get_messages(
        self,
        input: str,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": input},
        ]

    def invoke(self, input: str) -> LLMResponse:
        """Sends a text input to the OpenAI chat completion model
        and returns the response's content.

        Args:
            input (str): Text sent to the LLM

        Returns:
            LLMResponse: The response from OpenAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            response = self.client.chat.completions.create(
                messages=self.get_messages(input),  # type: ignore
                model=self.model_name,
                **self.model_params,
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content)
        except openai.OpenAIError as e:
            raise LLMGenerationError(e)

    async def ainvoke(self, input: str) -> LLMResponse:
        """Asynchronously sends a text input to the OpenAI chat
        completion model and returns the response's content.

        Args:
            input (str): Text sent to the LLM

        Returns:
            LLMResponse: The response from OpenAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            response = await self.async_client.chat.completions.create(
                messages=self.get_messages(input),  # type: ignore
                model=self.model_name,
                **self.model_params,
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content)
        except openai.OpenAIError as e:
            raise LLMGenerationError(e)


class AzureOpenAILLM(OpenAILLM):
    client_class: Type[openai.OpenAI] = openai.AzureOpenAI
    async_client_class: Type[openai.AsyncOpenAI] = openai.AsyncAzureOpenAI
