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

import warnings
from typing import TYPE_CHECKING, Any, Optional, Sequence

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.types import LLMMessage

from .base import LLMInterface
from neo4j_graphrag.utils.rate_limit import RateLimitHandler
from .types import (
    LLMResponse,
)

if TYPE_CHECKING:
    from ollama import Message


class OllamaLLM(LLMInterface):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ):
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "Could not import ollama Python client. "
                "Please install it with `pip install ollama`."
            )
        super().__init__(model_name, model_params, rate_limit_handler)
        self.ollama = ollama
        self.client = ollama.Client(
            **kwargs,
        )
        self.async_client = ollama.AsyncClient(
            **kwargs,
        )
        if "stream" in self.model_params:
            raise ValueError("Streaming is not supported by the OllamaLLM wrapper")
        # bug-fix with backward compatibility:
        # we mistakenly passed all "model_params" under the options argument
        # next two lines to be removed in 2.0
        if not any(
            key in self.model_params for key in ("options", "format", "keep_alive")
        ):
            warnings.warn(
                """Passing options directly without including them in an 'options' key is deprecated. Ie you must use model_params={"options": {"temperature": 0}}""",
                DeprecationWarning,
            )
            self.model_params = {"options": self.model_params}

    def get_messages(
        self,
        input: list[LLMMessage],
    ) -> Sequence[Message]:
        return [
            self.ollama.Message(**i)
            for i in input
        ]

    def _invoke(
        self,
        input: list[LLMMessage],
    ) -> LLMResponse:
        """Sends text to the LLM and returns a response.

        Args:
            input (str): The text to send to the LLM.

        Returns:
            LLMResponse: The response from the LLM.
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=self.get_messages(input),
                **self.model_params,
            )
            content = response.message.content or ""
            return LLMResponse(content=content)
        except self.ollama.ResponseError as e:
            raise LLMGenerationError(e)

    async def _ainvoke(
        self,
        input: list[LLMMessage],
    ) -> LLMResponse:
        """Asynchronously sends a text input to the OpenAI chat
        completion model and returns the response's content.

        Args:
            input (str): Text sent to the LLM.

        Returns:
            LLMResponse: The response from OpenAI.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        try:
            response = await self.async_client.chat(
                model=self.model_name,
                messages=self.get_messages(input),
                options=self.model_params,
            )
            content = response.message.content or ""
            return LLMResponse(content=content)
        except self.ollama.ResponseError as e:
            raise LLMGenerationError(e)
