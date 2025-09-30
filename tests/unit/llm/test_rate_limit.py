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

from typing import Any, Callable, Awaitable

import pytest
from unittest.mock import Mock
from tenacity import RetryError

from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
    NoOpRateLimitHandler,
    DEFAULT_RATE_LIMIT_HANDLER,
)
from neo4j_graphrag.exceptions import RateLimitError


def test_default_handler_retries_sync() -> None:
    call_count = 0

    def mock_func() -> None:
        nonlocal call_count
        call_count += 1
        raise RateLimitError("Rate limit exceeded")

    wrapped_func = DEFAULT_RATE_LIMIT_HANDLER.handle_sync(mock_func)

    with pytest.raises(RetryError):
        wrapped_func()

    assert call_count == 3


@pytest.mark.asyncio
async def test_default_handler_retries_async() -> None:
    call_count = 0

    async def mock_func() -> None:
        nonlocal call_count
        call_count += 1
        raise RateLimitError("Rate limit exceeded")

    wrapped_func = DEFAULT_RATE_LIMIT_HANDLER.handle_async(mock_func)

    with pytest.raises(RetryError):
        await wrapped_func()

    assert call_count == 3


def test_other_errors_pass_through_sync() -> None:
    call_count = 0

    def mock_func() -> None:
        nonlocal call_count
        call_count += 1
        raise ValueError("Some other error")

    wrapped_func = DEFAULT_RATE_LIMIT_HANDLER.handle_sync(mock_func)

    with pytest.raises(ValueError):
        wrapped_func()

    assert call_count == 1


@pytest.mark.asyncio
async def test_other_errors_pass_through_async() -> None:
    call_count = 0

    async def mock_func() -> None:
        nonlocal call_count
        call_count += 1
        raise ValueError("Some other error")

    wrapped_func = DEFAULT_RATE_LIMIT_HANDLER.handle_async(mock_func)

    with pytest.raises(ValueError):
        await wrapped_func()

    assert call_count == 1


def test_noop_handler_sync() -> None:
    def mock_func() -> str:
        return "test result"

    handler = NoOpRateLimitHandler()
    wrapped_func = handler.handle_sync(mock_func)

    assert wrapped_func() == "test result"
    assert wrapped_func is mock_func


@pytest.mark.asyncio
async def test_noop_handler_async() -> None:
    async def mock_func() -> str:
        return "async test result"

    handler = NoOpRateLimitHandler()
    wrapped_func = handler.handle_async(mock_func)

    assert await wrapped_func() == "async test result"
    assert wrapped_func is mock_func


def test_custom_handler_sync_retry_override() -> None:
    call_count = 0

    def mock_func() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError("Rate limit exceeded")
        return "success after custom retry"

    # Custom handler with single retry
    def custom_handle_sync(func: Callable[[], Any]) -> Callable[[], Any]:
        def wrapper() -> Any:
            try:
                return func()
            except RateLimitError:
                return func()  # Retry once

        return wrapper

    handler = Mock(spec=RateLimitHandler)
    handler.handle_sync = custom_handle_sync

    result = handler.handle_sync(mock_func)()
    assert result == "success after custom retry"
    assert call_count == 2


@pytest.mark.asyncio
async def test_custom_handler_async_retry_override() -> None:
    call_count = 0

    async def mock_func() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError("Rate limit exceeded")
        return "success after custom retry"

    # Custom handler with single retry
    def custom_handle_async(
        func: Callable[[], Awaitable[Any]],
    ) -> Callable[[], Awaitable[Any]]:
        async def wrapper() -> Any:
            try:
                return await func()
            except RateLimitError:
                return await func()  # Retry once

        return wrapper

    handler = Mock(spec=RateLimitHandler)
    handler.handle_async = custom_handle_async

    result = await handler.handle_async(mock_func)()
    assert result == "success after custom retry"
    assert call_count == 2
