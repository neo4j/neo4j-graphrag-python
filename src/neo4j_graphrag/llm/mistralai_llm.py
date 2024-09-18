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

import os
from typing import Any, Optional

from ..exceptions import LLMGenerationError
from .base import LLMInterface
from .types import LLMResponse

try:
    import mistralai
except ImportError:
    mistralai = None  # type: ignore


class MistralAILLM(LLMInterface):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """

        Args:
            model_name (str):
            model_params (str): Parameters like temperature and such  that will be
             passed to the model
            kwargs: All other parameters will be passed to the openai.OpenAI init.

        """
        if mistralai is None:
            raise ImportError(
                "Could not import Mistral Python client. "
                "Please install it with `pip install mistralai`."
            )
        super().__init__(model_name, model_params)
        api_key = os.getenv("MISTRAL_API_KEY", "")
        self.client = mistralai.Mistral(api_key=api_key, **kwargs)

    def get_messages(
        self,
        input: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "content": input,
                "role": "user",
            }
        ]

    def invoke(self, input: str) -> LLMResponse:
        """Sends a text input to the Mistral chat completion model
        and returns the response's content.

        Args:
            input (str): Text sent to the LLM

        Returns:
            LLMResponse: The response from MistralAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=self.get_messages(input),  # type: ignore
                **self.model_params,
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content)
        except mistralai.models.sdkerror.SDKError as e:
            raise LLMGenerationError(e)

    async def ainvoke(self, input: str) -> LLMResponse:
        """Asynchronously sends a text input to the MistralAI chat
        completion model and returns the response's content.

        Args:
            input (str): Text sent to the LLM

        Returns:
            LLMResponse: The response from OpenAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            response = await self.client.files.complete_async(
                model=self.model_name,
                messages=self.get_messages(input),
                **self.model_params,
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content)
        except mistralai.models.sdkerror.SDKError as e:
            raise LLMGenerationError(e)
