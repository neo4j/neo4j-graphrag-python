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

import functools
import logging
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional, TypeVar

from neo4j_graphrag.exceptions import RateLimitError

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
AF = TypeVar("AF", bound=Callable[..., Awaitable[Any]])


class RateLimitHandler(ABC):
    """Abstract base class for rate limit handling strategies."""

    @abstractmethod
    def handle_sync(self, func: F) -> F:
        """Apply rate limit handling to a synchronous function.

        Args:
            func: The function to wrap with rate limit handling.

        Returns:
            The wrapped function.
        """
        pass

    @abstractmethod
    def handle_async(self, func: AF) -> AF:
        """Apply rate limit handling to an asynchronous function.

        Args:
            func: The async function to wrap with rate limit handling.

        Returns:
            The wrapped async function.
        """
        pass


class NoOpRateLimitHandler(RateLimitHandler):
    """A no-op rate limit handler that does not apply any rate limiting."""

    def handle_sync(self, func: F) -> F:
        """Return the function unchanged."""
        return func

    def handle_async(self, func: AF) -> AF:
        """Return the async function unchanged."""
        return func


class RetryRateLimitHandler(RateLimitHandler):
    """Rate limit handler using exponential backoff retry strategy.

    This handler uses tenacity for retry logic with exponential backoff.
    Falls back to NoOpRateLimitHandler if tenacity is not available.

    Args:
        max_attempts: Maximum number of retry attempts. Defaults to 3.
        min_wait: Minimum wait time between retries in seconds. Defaults to 1.
        max_wait: Maximum wait time between retries in seconds. Defaults to 60.
        multiplier: Exponential backoff multiplier. Defaults to 2.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        min_wait: float = 1.0,
        max_wait: float = 60.0,
        multiplier: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.multiplier = multiplier

    def handle_sync(self, func: F) -> F:
        """Apply retry logic to a synchronous function."""

        @retry(
            retry=retry_if_exception_type(RateLimitError),
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(
                multiplier=self.multiplier,
                min=self.min_wait,
                max=self.max_wait,
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    def handle_async(self, func: AF) -> AF:
        """Apply retry logic to an asynchronous function."""

        @retry(
            retry=retry_if_exception_type(RateLimitError),
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(
                multiplier=self.multiplier,
                min=self.min_wait,
                max=self.max_wait,
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        return wrapper  # type: ignore


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if an exception is a rate limit error from any LLM provider.

    Args:
        exception: The exception to check.

    Returns:
        True if the exception indicates a rate limit error, False otherwise.
    """
    # Already converted to RateLimitError
    if isinstance(exception, RateLimitError):
        return True

    error_type = type(exception).__name__.lower()
    exception_str = str(exception).lower()

    # OpenAI - specific error type
    if error_type == "ratelimiterror":
        return True

    # Check for HTTP 429 status code (various providers)
    if hasattr(exception, "status_code") and getattr(exception, "status_code") == 429:
        return True

    if hasattr(exception, "response"):
        response = getattr(exception, "response")
        if hasattr(response, "status_code") and response.status_code == 429:
            return True

    # Provider-specific error types with message checks
    rate_limit_error_types = {
        "apierror": "too many requests",  # Anthropic, Cohere
        "sdkerror": "too many requests",  # MistralAI
        "responseerror": "too many requests",  # Ollama
        "responsevalidationerror": "resource exhausted",  # VertexAI (special case)
    }

    if error_type in rate_limit_error_types:
        required_message = rate_limit_error_types[error_type]
        return required_message in exception_str

    return False


def convert_to_rate_limit_error(exception: Exception) -> RateLimitError:
    """Convert a provider-specific rate limit exception to RateLimitError.

    Args:
        exception: The original exception from the LLM provider.

    Returns:
        A RateLimitError with the original exception message.
    """
    return RateLimitError(f"Rate limit exceeded: {exception}")


def rate_limit_handler(func: F) -> F:
    """Decorator to apply rate limit handling to synchronous methods.

    This decorator works with instance methods and uses the instance's rate limit handler.

    Args:
        func: The function to wrap with rate limit handling.

    Returns:
        The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Use instance handler or default
        active_handler = getattr(
            self, "_rate_limit_handler", DEFAULT_RATE_LIMIT_HANDLER
        )

        def inner_func() -> Any:
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                if is_rate_limit_error(e):
                    raise convert_to_rate_limit_error(e)
                raise

        return active_handler.handle_sync(inner_func)()

    return wrapper  # type: ignore


def async_rate_limit_handler(func: AF) -> AF:
    """Decorator to apply rate limit handling to asynchronous methods.

    This decorator works with instance methods and uses the instance's rate limit handler.

    Args:
        func: The async function to wrap with rate limit handling.

    Returns:
        The wrapped async function.
    """

    @functools.wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Use instance handler or default
        active_handler = getattr(
            self, "_rate_limit_handler", DEFAULT_RATE_LIMIT_HANDLER
        )

        async def inner_func() -> Any:
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                if is_rate_limit_error(e):
                    raise convert_to_rate_limit_error(e)
                raise

        return await active_handler.handle_async(inner_func)()

    return wrapper  # type: ignore


# Default rate limit handler instance
DEFAULT_RATE_LIMIT_HANDLER = RetryRateLimitHandler()
