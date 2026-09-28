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

# built-in dependencies
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Type,
    Union,
)

# 3rd party dependencies
from pydantic import BaseModel

# project dependencies
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMBase
from neo4j_graphrag.llm.types import (
    LLMResponse,
    LLMUsage,
)
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
)

if TYPE_CHECKING:
    from cohere import ChatMessages


# pylint: disable=redefined-builtin, arguments-differ, raise-missing-from, no-else-return, import-outside-toplevel
class CohereLLM(LLMBase):
    """Interface for large language models on the Cohere platform

    Args:
        model_name (str, optional): Name of the LLM to use. Defaults to "".
        model_params (Optional[dict], optional): Additional parameters passed to the model when text is sent to it. Defaults to None.
        system_instruction (Optional[str], optional): Additional instructions for setting the behavior and context for the model in a conversation. Defaults to None.
        rate_limit_handler (Optional[RateLimitHandler], optional): Handler for managing API rate limits. Defaults to None.
        **kwargs (Any): Arguments passed to the model when for the class is initialised. Defaults to None.

    Raises:
        LLMGenerationError: If there's an error generating the response from the model.

    Example:

    .. code-block:: python

        from neo4j_graphrag.llm import CohereLLM

        llm = CohereLLM(api_key="...")
        llm.invoke([{"role": "user", "content": "Say something"}])
    """

    def __init__(
        self,
        model_name: str = "",
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ) -> None:
        try:
            import cohere

            # Import the submodule rather than reaching for `cohere.core`: the
            # top-level package resolves attributes lazily and does not list
            # `core`, so the attribute chain raises AttributeError even though
            # the module is present and importable.
            from cohere.core.api_error import ApiError
        except ImportError:
            raise ImportError(
                """Could not import cohere python client.
                Please install it with `pip install "neo4j-graphrag[cohere]"`."""
            )
        LLMBase.__init__(
            self,
            model_name=model_name,
            model_params=model_params or {},
            rate_limit_handler=rate_limit_handler,
            **kwargs,
        )
        self.cohere = cohere
        self.cohere_api_error = ApiError

        self.client = cohere.ClientV2(**kwargs)
        self.async_client = cohere.AsyncClientV2(**kwargs)

    def _extract_text_content(self, content_items: Any) -> str:
        if not content_items:
            return ""
        text = getattr(content_items[0], "text", None)
        return text if isinstance(text, str) else ""

    # implementations
    def _build_request(
        self,
        messages: List[LLMMessage],
        *,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the Cohere chat request.

        response_format/unknown-role validation raise NotImplementedError/ValueError
        directly (not LLMGenerationError): these are caller usage errors, not SDK
        failures, and callers may want to catch them distinctly.
        """
        if response_format is not None:
            raise NotImplementedError(
                "CohereLLM does not currently support structured output"
            )
        return {
            "messages": self.get_messages(messages),
            "model": self.model_name,
        }

    def _call_sync(self, request: dict[str, Any]) -> Any:
        try:
            return self.client.chat(**request)
        except self.cohere_api_error as e:
            raise LLMGenerationError(e) from e

    async def _call_async(self, request: dict[str, Any]) -> Any:
        try:
            return await self.async_client.chat(**request)
        except self.cohere_api_error as e:
            raise LLMGenerationError(e) from e

    def _parse_response(self, raw_response: Any) -> LLMResponse:
        try:
            usage = None
            if raw_response.usage and raw_response.usage.tokens:
                input_tokens = (
                    int(raw_response.usage.tokens.input_tokens)
                    if raw_response.usage.tokens.input_tokens is not None
                    else None
                )
                output_tokens = (
                    int(raw_response.usage.tokens.output_tokens)
                    if raw_response.usage.tokens.output_tokens is not None
                    else None
                )
                usage = LLMUsage(
                    request_tokens=input_tokens,
                    response_tokens=output_tokens,
                    total_tokens=(input_tokens + output_tokens)
                    if (input_tokens is not None and output_tokens is not None)
                    else None,
                )
            return LLMResponse(
                content=(
                    raw_response.message.content[0].text
                    if raw_response.message.content
                    and hasattr(raw_response.message.content[0], "text")
                    else ""
                ),
                usage=usage,
            )
        except Exception as e:
            raise LLMGenerationError(e) from e

    def get_messages(
        self,
        input: list[LLMMessage],
    ) -> ChatMessages:
        """Converts a list of LLMMessage to ChatMessages for Cohere."""
        messages: ChatMessages = []
        for i in input:
            if i["role"] == "system":
                messages.append(self.cohere.SystemChatMessageV2(content=i["content"]))
            elif i["role"] == "user":
                messages.append(self.cohere.UserChatMessageV2(content=i["content"]))
            elif i["role"] == "assistant":
                messages.append(
                    self.cohere.AssistantChatMessageV2(content=i["content"])
                )
            else:
                raise ValueError(f"Unknown role: {i['role']}")
        return messages
