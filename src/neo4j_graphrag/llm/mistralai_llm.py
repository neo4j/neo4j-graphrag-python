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
from typing import Any, Optional, Union

from ..exceptions import LLMGenerationError
from .base import LLMInterface
from .types import LLMResponse

try:
    from mistralai import Mistral
    from mistralai.models.assistantmessage import AssistantMessage
    from mistralai.models.sdkerror import SDKError
    from mistralai.models.systemmessage import SystemMessage
    from mistralai.models.toolmessage import ToolMessage
    from mistralai.models.usermessage import UserMessage

    MessageType = Union[AssistantMessage, SystemMessage, ToolMessage, UserMessage]
except ImportError:
    Mistral = None  # type: ignore
    SDKError = None  # type: ignore


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
            model_params (str): Parameters like temperature and such that will be
             passed to the chat completions endpoint
            kwargs: All other parameters will be passed to the Mistral client.

        """
        if Mistral is None:
            raise ImportError(
                "Could not import Mistral Python client. "
                "Please install it with `pip install mistralai`."
            )
        super().__init__(model_name, model_params)
        api_key = kwargs.pop("api_key", None)
        if api_key is None:
            api_key = os.getenv("MISTRAL_API_KEY", "")
        self.client = Mistral(api_key=api_key, **kwargs)

    def get_messages(self, input: str) -> list[MessageType]:
        return [UserMessage(content=input)]

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
                messages=self.get_messages(input),
                **self.model_params,
            )
            if response is None or response.choices is None or not response.choices:
                content = ""
            else:
                content = response.choices[0].message.content or ""
            return LLMResponse(content=content)
        except SDKError as e:
            raise LLMGenerationError(e)

    async def ainvoke(self, input: str) -> LLMResponse:
        """Asynchronously sends a text input to the MistralAI chat
        completion model and returns the response's content.

        Args:
            input (str): Text sent to the LLM

        Returns:
            LLMResponse: The response from MistralAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            response = await self.client.chat.complete_async(
                model=self.model_name,
                messages=self.get_messages(input),
                **self.model_params,
            )
            if response is None or response.choices is None or not response.choices:
                content = ""
            else:
                content = response.choices[0].message.content or ""
            return LLMResponse(content=content)
        except SDKError as e:
            raise LLMGenerationError(e)
