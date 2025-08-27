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

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
)
from neo4j_graphrag.llm.types import (
    LLMResponse,
)
from neo4j_graphrag.types import LLMMessage

try:
    from mistralai import (
        Messages,
        UserMessage,
        AssistantMessage,
        SystemMessage,
        Mistral,
    )
    from mistralai.models.sdkerror import SDKError
except ImportError:
    Mistral = None  # type: ignore
    SDKError = None  # type: ignore
    Messages = None  # type: ignore


class MistralAILLM(LLMInterface):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ):
        """

        Args:
            model_name (str):
            model_params (str): Parameters like temperature and such that will be
             passed to the chat completions endpoint
            rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting. Defaults to retry with exponential backoff.
            kwargs: All other parameters will be passed to the Mistral client.

        """
        if Mistral is None:
            raise ImportError(
                """Could not import Mistral Python client.
                Please install it with `pip install "neo4j-graphrag[mistralai]"`."""
            )
        super().__init__(model_name, model_params, rate_limit_handler)
        api_key = kwargs.pop("api_key", None)
        if api_key is None:
            api_key = os.getenv("MISTRAL_API_KEY", "")
        self.client = Mistral(api_key=api_key, **kwargs)

    def get_messages(
        self,
        input: list[LLMMessage],
    ) -> list[Messages]:
        messages: list[Messages] = []
        for m in input:
            if m["role"] == "system":
                messages.append(SystemMessage(content=m["content"]))
                continue
            if m["role"] == "user":
                messages.append(UserMessage(content=m["content"]))
                continue
            if m["role"] == "assistant":
                messages.append(AssistantMessage(content=m["content"]))
                continue
        return messages

    def _invoke(
        self,
        input: list[LLMMessage],
    ) -> LLMResponse:
        """Sends a text input to the Mistral chat completion model
        and returns the response's content.

        Args:
            input (str): Text sent to the LLM.

        Returns:
            LLMResponse: The response from MistralAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            messages = self.get_messages(input)
            response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                **self.model_params,
            )
            content: str = ""
            if response and response.choices:
                possible_content = response.choices[0].message.content
                if isinstance(possible_content, str):
                    content = possible_content
            return LLMResponse(content=content)
        except SDKError as e:
            raise LLMGenerationError(e)

    async def _ainvoke(
        self,
        input: list[LLMMessage],
    ) -> LLMResponse:
        """Asynchronously sends a text input to the MistralAI chat
        completion model and returns the response's content.

        Args:
            input (str): Text sent to the LLM.

        Returns:
            LLMResponse: The response from MistralAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            messages = self.get_messages(input)
            response = await self.client.chat.complete_async(
                model=self.model_name,
                messages=messages,
                **self.model_params,
            )
            content: str = ""
            if response and response.choices:
                possible_content = response.choices[0].message.content
                if isinstance(possible_content, str):
                    content = possible_content
            return LLMResponse(content=content)
        except SDKError as e:
            raise LLMGenerationError(e)
