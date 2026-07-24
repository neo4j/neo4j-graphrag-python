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
from typing import AsyncGenerator, Generator

import warnings

import httpx
import pytest
import pytest_asyncio

from neo4j_graphrag.llm.utils import (
    legacy_inputs_to_messages,
    split_http_client_kwargs,
    system_instruction_from_messages,
)
from neo4j_graphrag.message_history import InMemoryMessageHistory
from neo4j_graphrag.types import LLMMessage


@pytest.fixture
def httpx_sync_client() -> Generator[httpx.Client, None, None]:
    client = httpx.Client()
    yield client
    client.close()


@pytest_asyncio.fixture
async def httpx_async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    client = httpx.AsyncClient()
    yield client
    await client.aclose()


def test_system_instruction_from_messages_found() -> None:
    messages = [
        LLMMessage(role="system", content="You are helpful"),
        LLMMessage(role="user", content="hi"),
    ]
    assert system_instruction_from_messages(messages) == "You are helpful"


def test_system_instruction_from_messages_not_found() -> None:
    messages = [LLMMessage(role="user", content="hi")]
    assert system_instruction_from_messages(messages) is None


def test_legacy_inputs_with_message_history_instance() -> None:
    history = InMemoryMessageHistory()
    history.add_message(LLMMessage(role="user", content="previous"))
    result = legacy_inputs_to_messages("follow-up", message_history=history)
    assert result[-1]["content"] == "follow-up"
    assert result[0]["content"] == "previous"


def test_legacy_inputs_system_instruction_conflict_warns() -> None:
    messages = [LLMMessage(role="system", content="existing")]
    with pytest.warns(UserWarning, match="system_instruction provided but ignored"):
        legacy_inputs_to_messages(
            "hi", message_history=messages, system_instruction="new"
        )


def test_legacy_inputs_prompt_as_list() -> None:
    prompt_list = [LLMMessage(role="user", content="hello")]
    result = legacy_inputs_to_messages(prompt_list)
    assert result == prompt_list


def test_legacy_inputs_prompt_as_message_history() -> None:
    history = InMemoryMessageHistory()
    history.add_message(LLMMessage(role="user", content="from history"))
    result = legacy_inputs_to_messages(history)
    assert len(result) == 1
    assert result[0]["content"] == "from history"


def test_split_http_client_kwargs_no_http_client() -> None:
    sync_kwargs, async_kwargs = split_http_client_kwargs({"api_key": "sk-test"})
    assert sync_kwargs == {"api_key": "sk-test"}
    assert async_kwargs == {"api_key": "sk-test"}
    assert "http_client" not in sync_kwargs
    assert "http_client" not in async_kwargs


def test_split_http_client_kwargs_routes_sync_client(
    httpx_sync_client: httpx.Client,
) -> None:
    sync_kwargs, async_kwargs = split_http_client_kwargs(
        {"api_key": "sk-test", "http_client": httpx_sync_client}
    )
    assert sync_kwargs["http_client"] is httpx_sync_client
    assert sync_kwargs["api_key"] == "sk-test"
    assert "http_client" not in async_kwargs
    assert async_kwargs["api_key"] == "sk-test"


@pytest.mark.asyncio
async def test_split_http_client_kwargs_routes_async_client(
    httpx_async_client: httpx.AsyncClient,
) -> None:
    sync_kwargs, async_kwargs = split_http_client_kwargs(
        {"api_key": "sk-test", "http_client": httpx_async_client}
    )
    assert async_kwargs["http_client"] is httpx_async_client
    assert async_kwargs["api_key"] == "sk-test"
    assert "http_client" not in sync_kwargs
    assert sync_kwargs["api_key"] == "sk-test"


def test_split_http_client_kwargs_warns_on_client_with_base_url() -> None:
    client = httpx.Client(base_url="https://my-endpoint.example.com")
    try:
        with pytest.warns(UserWarning, match="base_url configured on the provided"):
            sync_kwargs, _ = split_http_client_kwargs({"http_client": client})
        # the client is still routed; only the base_url is flagged
        assert sync_kwargs["http_client"] is client
    finally:
        client.close()


@pytest.mark.asyncio
async def test_split_http_client_kwargs_warns_on_async_client_with_base_url() -> None:
    client = httpx.AsyncClient(base_url="https://my-endpoint.example.com")
    try:
        with pytest.warns(UserWarning, match="base_url configured on the provided"):
            _, async_kwargs = split_http_client_kwargs({"http_client": client})
        assert async_kwargs["http_client"] is client
    finally:
        await client.aclose()


def test_split_http_client_kwargs_no_warning_without_base_url(
    httpx_sync_client: httpx.Client,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        split_http_client_kwargs({"http_client": httpx_sync_client})


def test_split_http_client_kwargs_invalid_type_warns_and_drops() -> None:
    with pytest.warns(UserWarning, match="Invalid http_client type"):
        sync_kwargs, async_kwargs = split_http_client_kwargs(
            {"api_key": "sk-test", "http_client": object()}
        )
    assert "http_client" not in sync_kwargs
    assert "http_client" not in async_kwargs
    assert sync_kwargs["api_key"] == "sk-test"
    assert async_kwargs["api_key"] == "sk-test"


def test_split_http_client_kwargs_does_not_mutate_input(
    httpx_sync_client: httpx.Client,
) -> None:
    original: dict[str, object] = {
        "api_key": "sk-test",
        "http_client": httpx_sync_client,
    }
    original_copy = dict(original)
    split_http_client_kwargs(original)
    assert original == original_copy
