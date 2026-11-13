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

import asyncio
import os
from typing import TYPE_CHECKING, Any, Optional

from neo4j_graphrag.exceptions import (
    DecisionModelError,
    RateLimitError,
    RetryableError,
)
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
    RetryRateLimitHandler,
)
from neo4j_graphrag.utils.rate_limit import (
    async_rate_limit_handler as async_rate_limit_handler_decorator,
)
from neo4j_graphrag.utils.rate_limit import (
    rate_limit_handler as rate_limit_handler_decorator,
)

from .base import DecisionModelInterface, DecisionState
from .types import DecisionResponse, DecisionUsage

if TYPE_CHECKING:
    import httpx

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

# The decision API accepts several named questions in one request. One is enough here.
QUESTION_ID = "next_step"

# Besides 429, these statuses are retried. They mean a timeout, or a provider that is
# overloaded or can't be reached.
RETRYABLE_STATUS_CODES = {408, 500, 502, 503, 504, 524, 529}

DEFAULT_DECISION_RATE_LIMIT_HANDLER = RetryRateLimitHandler(
    max_attempts=5, retryable_exceptions=(RetryableError,)
)


# pylint: disable=redefined-builtin, raise-missing-from, import-outside-toplevel
class OpenRouterDecisionModel(DecisionModelInterface):
    """Decision models served by OpenRouter's decision API.

    OpenRouter's decision API (``/api/alpha/decisions``) is still in alpha, so the format
    of its requests and responses may change.

    Example:

    .. code-block:: python

        from neo4j_graphrag.decision import OpenRouterDecisionModel

        model = OpenRouterDecisionModel(
            "liquid/d1-ultra",
            provider={"order": ["liquid"], "allow_fallbacks": False},
        )
        response = model.decide(
            state={"question": "Who directed Inception?"},
            options={"0": "follow directed_by", "1": "follow starred_actors"},
            instructions="Choose the connection to follow to answer the question.",
        )
        response.choice  # "0"
        response.probabilities  # {"0": 0.97, "1": 0.03}

    Args:
        model_name (str): The OpenRouter model name, for example ``"liquid/d1-ultra"``.
        api_key (Optional[str]): The OpenRouter API key. If it isn't given, the
            ``OPENROUTER_API_KEY`` environment variable is used.
        endpoint_url (str): The full URL of the decision endpoint. Defaults to OpenRouter's.
        provider (Optional[dict[str, Any]]): OpenRouter's provider routing settings, which
            are sent unchanged, for example ``{"order": ["liquid"], "allow_fallbacks": False}``.
        max_options (int): The largest number of options that the model accepts in one
            call. Defaults to 255.
        timeout (float): The timeout for one HTTP request, in seconds. Defaults to 60.
        rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting. By
            default, a request is tried up to 5 times, with exponential backoff, after
            HTTP 429, a timeout, a network error or a server error. A request that timed
            out may still have been answered, so a retry can be billed twice.

    Raises:
        RateLimitError: On HTTP 429. It is a kind of
            :class:`~neo4j_graphrag.exceptions.RetryableError`.
        RetryableError: On a timeout, a network error, or HTTP 408 or 5xx. When a rate
            limit handler stops retrying, it raises ``tenacity.RetryError``, as it does for
            the library's LLMs.
        DecisionModelError: On any other failed request, an unexpected response, or a
            choice that is not one of the options.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        endpoint_url: str = OPENROUTER_DECISIONS_URL,
        provider: Optional[dict[str, Any]] = None,
        max_options: int = 255,
        timeout: float = 60.0,
        rate_limit_handler: Optional[RateLimitHandler] = None,
    ):
        try:
            import httpx
        except ImportError:
            raise ImportError(
                """Could not import httpx Python client.
                Please install it with `pip install "neo4j-graphrag[openrouter]"`."""
            )
        super().__init__(
            model_name=model_name,
            max_options=max_options,
            rate_limit_handler=rate_limit_handler
            or DEFAULT_DECISION_RATE_LIMIT_HANDLER,
        )
        api_key = (api_key or os.environ.get("OPENROUTER_API_KEY") or "").strip()
        if not api_key:
            raise ValueError(
                "An OpenRouter API key is required. Pass api_key, or set OPENROUTER_API_KEY."
            )
        self.httpx = httpx
        self.endpoint_url = endpoint_url
        self.provider = provider
        headers = {"Authorization": f"Bearer {api_key}"}
        self.client = httpx.Client(headers=headers, timeout=timeout)
        self.async_client = httpx.AsyncClient(headers=headers, timeout=timeout)

    @rate_limit_handler_decorator
    def decide(
        self,
        state: DecisionState,
        options: dict[str, str],
        instructions: str,
    ) -> DecisionResponse:
        request = self._build_request(state, options, instructions)
        try:
            response = self.client.post(self.endpoint_url, json=request)
        except self.httpx.HTTPError as e:
            # Not chained, because the original error's text can contain the request headers
            raise self._request_error(e) from None
        return self._parse_response(response, options)

    @async_rate_limit_handler_decorator
    async def adecide(
        self,
        state: DecisionState,
        options: dict[str, str],
        instructions: str,
    ) -> DecisionResponse:
        request = self._build_request(state, options, instructions)
        try:
            response = await self.async_client.post(self.endpoint_url, json=request)
        except self.httpx.HTTPError as e:
            # Not chained, because the original error's text can contain the request headers
            raise self._request_error(e) from None
        return self._parse_response(response, options)

    def close(self) -> None:
        """Close both HTTP clients.

        Don't call it from inside a running event loop. Use ``await aclose()`` there.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.aclose())
            finally:
                loop.close()
        else:
            raise RuntimeError(
                "Cannot call close() from a running async context. "
                "Use 'async with' or 'await aclose()' instead."
            )

    async def aclose(self) -> None:
        """Close both HTTP clients."""
        self.client.close()
        await self.async_client.aclose()

    def __enter__(self) -> "OpenRouterDecisionModel":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "OpenRouterDecisionModel":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    def _build_request(
        self, state: DecisionState, options: dict[str, str], instructions: str
    ) -> dict[str, Any]:
        if not options:
            raise DecisionModelError("A decision needs at least one option.")
        if len(options) > self.max_options:
            raise DecisionModelError(
                f"{len(options)} options exceed the model's limit of {self.max_options}."
            )
        request: dict[str, Any] = {
            "model": self.model_name,
            "state": state,
            "questions": {
                QUESTION_ID: {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": options,
                }
            },
        }
        if self.provider is not None:
            request["provider"] = self.provider
        return request

    def _request_error(self, error: httpx.HTTPError) -> Exception:
        """Turn an HTTP error into an exception that is retried, or one that is not.

        Network errors and timeouts are retried. For any other error, only the error type
        goes into the message, because the error's text can contain the request headers,
        and so the API key.
        """
        if isinstance(
            error,
            (
                self.httpx.TimeoutException,
                self.httpx.NetworkError,
                self.httpx.RemoteProtocolError,
            ),
        ):
            return RetryableError(
                f"OpenRouter decision request failed: {type(error).__name__}: {error}"
            )
        return DecisionModelError(
            f"OpenRouter decision request failed: {type(error).__name__}"
        )

    def _parse_response(
        self, response: httpx.Response, options: dict[str, str]
    ) -> DecisionResponse:
        status = response.status_code
        if status != 200:
            message = (
                f"OpenRouter decision request failed with HTTP {status}: "
                f"{response.text[:500]}"
            )
            if status == 429:
                raise RateLimitError(message)
            if status in RETRYABLE_STATUS_CODES:
                raise RetryableError(message)
            raise DecisionModelError(message)
        try:
            body = response.json()
            answer = body["answers"][QUESTION_ID]
            usage = body.get("usage") or {}
            decision = DecisionResponse(
                choice=str(answer["choice"]),
                probabilities={
                    str(key): float(p) for key, p in answer["probabilities"].items()
                },
                confidence=answer.get("confidence"),
                model=body.get("model"),
                usage=DecisionUsage(
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    cost=usage.get("cost"),
                ),
            )
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            # pydantic's ValidationError is a ValueError
            raise DecisionModelError(
                f"Unexpected decision response: {response.text[:500]}"
            ) from e
        if decision.choice not in options:
            raise DecisionModelError(
                f"The model chose {decision.choice!r}, which is not one of the options."
            )
        return decision
