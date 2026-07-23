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
from typing import Any, Union, Optional

from pydantic import TypeAdapter

from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


def system_instruction_from_messages(messages: list[LLMMessage]) -> str | None:
    """Extracts the system instruction from a list of LLMMessage, if present."""
    for message in messages:
        if message["role"] == "system":
            return message["content"]
    return None


llm_messages_adapter = TypeAdapter(list[LLMMessage])


def legacy_inputs_to_messages(
    prompt: Union[str, list[LLMMessage], MessageHistory],
    message_history: Optional[Union[list[LLMMessage], MessageHistory]] = None,
    system_instruction: Optional[str] = None,
) -> list[LLMMessage]:
    """Converts legacy prompt and message history inputs to a unified list of LLMMessage."""
    if message_history:
        if isinstance(message_history, MessageHistory):
            messages = message_history.messages
        else:  # list[LLMMessage]
            messages = llm_messages_adapter.validate_python(message_history)
    else:
        messages = []
    if system_instruction is not None:
        if system_instruction_from_messages(messages) is not None:
            warnings.warn(
                "system_instruction provided but ignored as the message history already contains a system message",
                UserWarning,
            )
        else:
            messages.insert(
                0,
                LLMMessage(
                    role="system",
                    content=system_instruction,
                ),
            )

    if isinstance(prompt, str):
        messages.append(LLMMessage(role="user", content=prompt))
        return messages
    if isinstance(prompt, list):
        messages.extend(prompt)
        return messages
    # prompt is a MessageHistory instance
    messages.extend(prompt.messages)
    return messages


def split_http_client_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Splits a shared ``kwargs`` dict into separate sync/async constructor kwargs,
    routing an optional ``http_client`` to whichever SDK client it matches the type of.

    Several provider integrations (``AnthropicLLM``, ``OpenAILLM``, ``AzureOpenAILLM``)
    build both a sync and an async SDK client from a single constructor ``**kwargs``
    dict. Most kwargs (``api_key``, ``max_retries``, ``default_headers``, ...) are safe
    to share as-is, but ``http_client`` is not: the sync client needs an
    ``httpx.Client``, the async client needs an ``httpx.AsyncClient``, and passing the
    wrong type to either raises or silently misbehaves depending on SDK version.

    This pops ``http_client`` out of *kwargs* and returns two independent copies of the
    remaining kwargs, with ``http_client`` added back to only the copy whose SDK client
    it matches. If ``http_client`` doesn't match either expected type, a warning is
    emitted and it is dropped from both, falling back to each SDK's default transport.

    Args:
        kwargs: The shared constructor kwargs, as passed by a caller to e.g.
            ``AnthropicLLM(...)``. Not mutated.

    Returns:
        A ``(sync_kwargs, async_kwargs)`` tuple, each a shallow copy of *kwargs* minus
        ``http_client``, with ``http_client`` reinstated in whichever of the two it
        belongs to.
    """
    kwargs = dict(kwargs)
    http_client = kwargs.pop("http_client", None)
    sync_kwargs = kwargs.copy()
    async_kwargs = kwargs.copy()
    if httpx is not None and isinstance(http_client, (httpx.Client, httpx.AsyncClient)):
        if str(http_client.base_url):
            # stacklevel=3 attributes the warning to the caller of the LLM
            # constructor, not to the constructor's own call into this helper.
            warnings.warn(
                "The base_url configured on the provided http_client is ignored: "
                "the SDK builds request URLs from its own base_url and uses the "
                "http_client as transport only. Pass base_url to the LLM "
                "constructor instead.",
                stacklevel=3,
            )
        if isinstance(http_client, httpx.Client):
            sync_kwargs["http_client"] = http_client
        else:
            async_kwargs["http_client"] = http_client
    elif http_client is not None:
        # stacklevel=3 attributes the warning to the caller of the LLM
        # constructor, not to the constructor's own call into this helper.
        warnings.warn(
            f"Invalid http_client type (got {type(http_client)}, expected httpx.Client or httpx.AsyncClient). Using default client.",
            stacklevel=3,
        )
    return sync_kwargs, async_kwargs
