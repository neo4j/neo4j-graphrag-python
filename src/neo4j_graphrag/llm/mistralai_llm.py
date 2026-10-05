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
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import BaseLLM
from neo4j_graphrag.llm.types import (
    LLMResponse,
    LLMUsage,
)
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
)

try:
    # mistralai 2.x turned the top-level package into a namespace and moved
    # everything under mistralai.client; there are no top-level exports left.
    from mistralai.client import Mistral
    from mistralai.client.errors import SDKError
    from mistralai.client.models import (
        AssistantMessage,
    )
    from mistralai.client.models import (
        # v1's `Messages` union; renamed, same meaning.
        ChatCompletionRequestMessage as Messages,
    )
    from mistralai.client.models import (
        SystemMessage as MistralSystemMessage,
    )
    from mistralai.client.models import (
        UserMessage as MistralUserMessage,
    )
except ImportError:
    Mistral = None  # type: ignore[assignment, misc]


# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from, no-else-return
class MistralAILLM(BaseLLM):
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
            model_params (str): Parameters like temperature that will be
             passed to the chat completions endpoint
            rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting. Defaults to retry with exponential backoff.
            kwargs: All other parameters will be passed to the Mistral client.

        """
        if Mistral is None:
            raise ImportError(
                """Could not import Mistral Python client.
                Please install it with `pip install "neo4j-graphrag[mistralai]"`."""
            )
        BaseLLM.__init__(
            self,
            model_name=model_name,
            model_params=model_params or {},
            rate_limit_handler=rate_limit_handler,
            **kwargs,
        )
        api_key = kwargs.pop("api_key", None)
        if api_key is None:
            api_key = os.getenv("MISTRAL_API_KEY", "")
        self.client = Mistral(api_key=api_key, **kwargs)

    # implementations
    def _build_request(
        self,
        messages: List[LLMMessage],
        *,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the kwargs for the Mistral chat completion endpoint.

        Shared by the sync and async invoke paths, which differ only in how they
        call the SDK.
        """
        if response_format is not None:
            raise NotImplementedError(
                "MistralAILLM does not currently support structured output"
            )
        return {
            "model": self.model_name,
            "messages": self.get_messages(messages),
            **self.model_params,
            **kwargs,
        }

    @staticmethod
    def _parse_response_content(response: Any) -> tuple[str, Optional[LLMUsage]]:
        """Pull the content and token usage out of a chat completion."""
        content = ""
        usage = None
        if response and response.choices:
            # mistralai 2.x types `message` as optional, so a choice can arrive
            # without one - a filtered or truncated completion.
            message = response.choices[0].message
            possible_content = message.content if message else None
            if isinstance(possible_content, str):
                content = possible_content
        if response and response.usage:
            usage = LLMUsage(
                request_tokens=response.usage.prompt_tokens,
                response_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        return content, usage

    def _parse_response(self, raw_response: Any) -> LLMResponse:
        content, usage = self._parse_response_content(raw_response)
        return LLMResponse(content=content, usage=usage)

    def _call_sync(self, request: dict[str, Any]) -> Any:
        try:
            return self.client.chat.complete(**request)
        except SDKError as e:
            raise LLMGenerationError(e) from e

    async def _call_async(self, request: dict[str, Any]) -> Any:
        try:
            return await self.client.chat.complete_async(**request)
        except SDKError as e:
            raise LLMGenerationError(e) from e

    async def aclose(self) -> None:
        # mistralai 2.x dropped close()/aclose() in favour of the context
        # manager protocol, so drive that directly. Both are safe on a client
        # that was never entered, and safe to call twice.
        self.client.__exit__(None, None, None)
        await self.client.__aexit__(None, None, None)

    def get_messages(
        self,
        input: list[LLMMessage],
    ) -> list[Messages]:
        """Constructs the message list for the Mistral chat completion model."""
        messages: list[Messages] = []
        for m in input:
            if m["role"] == "system":
                messages.append(MistralSystemMessage(content=m["content"]))
                continue
            if m["role"] == "user":
                messages.append(MistralUserMessage(content=m["content"]))
                continue
            if m["role"] == "assistant":
                messages.append(AssistantMessage(content=m["content"]))
                continue
            raise ValueError(f"Unknown role: {m['role']}")
        return messages
