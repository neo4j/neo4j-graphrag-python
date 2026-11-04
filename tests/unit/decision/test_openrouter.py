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
import traceback
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import tenacity

import neo4j_graphrag.decision as decision_module
from neo4j_graphrag.decision import (
    DecisionModelInterface,
    DecisionResponse,
    DecisionUsage,
    OpenRouterDecisionModel,
)
from neo4j_graphrag.decision.openrouter import (
    DEFAULT_DECISION_RATE_LIMIT_HANDLER,
    OPENROUTER_DECISIONS_URL,
)
from neo4j_graphrag.exceptions import (
    DecisionModelError,
    RateLimitError,
    RetryableError,
)
from neo4j_graphrag.utils.rate_limit import (
    NoOpRateLimitHandler,
    RetryRateLimitHandler,
)

OPTIONS = {"0": "STOP: the answer is: Inception", "1": "follow directed_by (outgoing)"}

RESPONSE_BODY = {
    "model": "liquid/d1-ultra-20261015",
    "answers": {
        "next_step": {
            "type": "choice",
            "choice": "1",
            "probabilities": {"0": 0.07, "1": 0.93},
            "confidence": 0.89,
        }
    },
    "usage": {"input_tokens": 114, "output_tokens": 0, "cost": 4.788e-06},
    "id": "gen-dec-1",
    "provider": "Liquid",
}


def make_response(status: int = 200, body: Any = RESPONSE_BODY) -> httpx.Response:
    request = httpx.Request("POST", OPENROUTER_DECISIONS_URL)
    if isinstance(body, str):
        return httpx.Response(status, text=body, request=request)
    return httpx.Response(status, json=body, request=request)


def make_model(
    **kwargs: Any,
) -> tuple[OpenRouterDecisionModel, MagicMock, MagicMock]:
    """The model, with mocks in place of its sync and async HTTP clients."""
    kwargs.setdefault("rate_limit_handler", NoOpRateLimitHandler())
    model = OpenRouterDecisionModel("liquid/d1-ultra", api_key="test-key", **kwargs)
    client, async_client = MagicMock(), MagicMock()
    setattr(model, "client", client)
    setattr(model, "async_client", async_client)
    return model, client, async_client


def test_openrouter_missing_dependency() -> None:
    with patch.dict("sys.modules", {"httpx": None}):
        with pytest.raises(ImportError, match="neo4j-graphrag\\[openrouter\\]"):
            OpenRouterDecisionModel("liquid/d1-ultra", api_key="test-key")


def test_openrouter_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterDecisionModel("liquid/d1-ultra")


def test_openrouter_reads_api_key_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    model = OpenRouterDecisionModel("liquid/d1-ultra")
    assert model.client.headers["Authorization"] == "Bearer env-key"
    assert model.max_options == 255
    assert isinstance(model, DecisionModelInterface)


def test_openrouter_rejects_max_options_below_two() -> None:
    with pytest.raises(ValueError, match="max_options"):
        OpenRouterDecisionModel("liquid/d1-ultra", api_key="test-key", max_options=1)


def test_openrouter_decide_happy_path() -> None:
    provider = {"order": ["liquid"], "allow_fallbacks": False}
    model, client, async_client = make_model(provider=provider)
    client.post.return_value = make_response()

    response = model.decide(
        {"question": "Who directed Inception?"}, OPTIONS, "Pick one."
    )

    assert response == DecisionResponse(
        choice="1",
        probabilities={"0": 0.07, "1": 0.93},
        confidence=0.89,
        model="liquid/d1-ultra-20261015",
        usage=DecisionUsage(input_tokens=114, output_tokens=0, cost=4.788e-06),
    )
    client.post.assert_called_once_with(
        OPENROUTER_DECISIONS_URL,
        json={
            "model": "liquid/d1-ultra",
            "state": {"question": "Who directed Inception?"},
            "questions": {
                "next_step": {
                    "type": "choice",
                    "instructions": "Pick one.",
                    "criteria": OPTIONS,
                }
            },
            "provider": provider,
        },
    )


def test_openrouter_decide_without_provider_omits_it() -> None:
    model, client, async_client = make_model()
    client.post.return_value = make_response()
    model.decide("state", OPTIONS, "Pick one.")
    assert "provider" not in client.post.call_args.kwargs["json"]


def test_openrouter_decide_without_usage() -> None:
    body = {"answers": RESPONSE_BODY["answers"]}
    model, client, async_client = make_model()
    client.post.return_value = make_response(body=body)
    response = model.decide("state", OPTIONS, "Pick one.")
    assert response.model is None
    assert response.usage is not None and response.usage.cost is None


@pytest.mark.parametrize(
    "status, error",
    [
        (429, RateLimitError),
        (503, RetryableError),
        (408, RetryableError),
        (400, DecisionModelError),
        (401, DecisionModelError),
    ],
)
def test_openrouter_decide_http_errors(status: int, error: type[Exception]) -> None:
    model, client, async_client = make_model()
    client.post.return_value = make_response(status, body="error body")
    with pytest.raises(error, match=f"HTTP {status}"):
        model.decide("state", OPTIONS, "Pick one.")


def test_openrouter_decide_transport_error_is_retryable() -> None:
    model, client, async_client = make_model()
    client.post.side_effect = httpx.ConnectTimeout("timed out")
    with pytest.raises(RetryableError, match="ConnectTimeout"):
        model.decide("state", OPTIONS, "Pick one.")


def test_openrouter_decide_local_error_is_not_retried_and_hides_headers() -> None:
    model, client, async_client = make_model(
        rate_limit_handler=RetryRateLimitHandler(
            max_attempts=3,
            min_wait=0,
            max_wait=0,
            retryable_exceptions=(RetryableError,),
        )
    )
    client.post.side_effect = httpx.LocalProtocolError(
        "Illegal header value b'Bearer secret-key\\r'"
    )
    with pytest.raises(DecisionModelError) as exc_info:
        model.decide("state", OPTIONS, "Pick one.")
    assert str(exc_info.value) == (
        "OpenRouter decision request failed: LocalProtocolError"
    )
    # The header isn't in the traceback either, which logging.exception would print.
    assert "secret-key" not in "".join(traceback.format_exception(exc_info.value))
    assert client.post.call_count == 1


def test_openrouter_default_rate_limit_handler() -> None:
    model = OpenRouterDecisionModel("liquid/d1-ultra", api_key="test-key")
    handler = model._rate_limit_handler
    assert handler is DEFAULT_DECISION_RATE_LIMIT_HANDLER
    assert isinstance(handler, RetryRateLimitHandler)
    assert handler.max_attempts == 5
    assert handler.retryable_exceptions == (RetryableError,)


def test_openrouter_decide_gives_up_after_retries() -> None:
    model, client, async_client = make_model(
        rate_limit_handler=RetryRateLimitHandler(
            max_attempts=2,
            min_wait=0,
            max_wait=0,
            retryable_exceptions=(RetryableError,),
        )
    )
    client.post.return_value = make_response(429, body="slow down")
    with pytest.raises(tenacity.RetryError):
        model.decide("state", OPTIONS, "Pick one.")
    assert client.post.call_count == 2


def test_openrouter_strips_api_key() -> None:
    model = OpenRouterDecisionModel("liquid/d1-ultra", api_key=" key-from-a-file\r\n")
    assert model.client.headers["Authorization"] == "Bearer key-from-a-file"


def test_openrouter_decide_retries_rate_limits() -> None:
    model, client, async_client = make_model(
        rate_limit_handler=RetryRateLimitHandler(
            max_attempts=3,
            min_wait=0,
            max_wait=0,
            retryable_exceptions=(RetryableError,),
        )
    )
    client.post.side_effect = [
        make_response(429, body="slow down"),
        make_response(502, body="bad gateway"),
        make_response(),
    ]
    assert model.decide("state", OPTIONS, "Pick one.").choice == "1"
    assert client.post.call_count == 3


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        {"answers": {}},
        {"answers": {"next_step": {"choice": "1"}}},
        {"answers": {"next_step": {"choice": "1", "probabilities": ["0.5"]}}},
        {
            "answers": {
                "next_step": {
                    "choice": "1",
                    "probabilities": {"1": 1.0},
                    "confidence": "high",
                }
            }
        },
        {
            "answers": {"next_step": {"choice": "1", "probabilities": {"1": 1.0}}},
            "usage": [1],
        },
        {
            "answers": {"next_step": {"choice": "1", "probabilities": {"1": 1.0}}},
            "usage": {"input_tokens": "n/a"},
        },
    ],
)
def test_openrouter_decide_malformed_response(body: Any) -> None:
    model, client, async_client = make_model()
    client.post.return_value = make_response(body=body)
    with pytest.raises(DecisionModelError, match="Unexpected decision response"):
        model.decide("state", OPTIONS, "Pick one.")


def test_openrouter_decide_choice_not_an_option() -> None:
    body = {"answers": {"next_step": {"choice": "7", "probabilities": {"7": 1.0}}}}
    model, client, async_client = make_model()
    client.post.return_value = make_response(body=body)
    with pytest.raises(DecisionModelError, match="not one of the options"):
        model.decide("state", OPTIONS, "Pick one.")


def test_openrouter_decide_integer_choice() -> None:
    body = {
        "answers": {"next_step": {"choice": 0, "probabilities": {"0": 0.6, "1": 0.4}}}
    }
    model, client, async_client = make_model()
    client.post.return_value = make_response(body=body)
    assert model.decide("state", OPTIONS, "Pick one.").choice == "0"


def test_openrouter_decide_validates_options() -> None:
    model, client, async_client = make_model(max_options=2)
    with pytest.raises(DecisionModelError, match="at least one option"):
        model.decide("state", {}, "Pick one.")
    with pytest.raises(DecisionModelError, match="exceed the model's limit of 2"):
        model.decide("state", {"0": "a", "1": "b", "2": "c"}, "Pick one.")
    client.post.assert_not_called()


@pytest.mark.asyncio
async def test_openrouter_adecide_happy_path() -> None:
    model, client, async_client = make_model()
    async_client.post = AsyncMock(return_value=make_response())
    response = await model.adecide({"question": "q"}, OPTIONS, "Pick one.")
    assert response.choice == "1"
    assert response.probabilities == {"0": 0.07, "1": 0.93}


@pytest.mark.asyncio
async def test_openrouter_adecide_transport_error_is_retryable() -> None:
    model, client, async_client = make_model()
    async_client.post = AsyncMock(side_effect=httpx.ReadError("reset"))
    with pytest.raises(RetryableError, match="ReadError"):
        await model.adecide({"question": "q"}, OPTIONS, "Pick one.")


@pytest.mark.asyncio
async def test_openrouter_adecide_http_errors() -> None:
    model, client, async_client = make_model()
    async_client.post = AsyncMock(return_value=make_response(429, body="slow down"))
    with pytest.raises(RateLimitError, match="HTTP 429"):
        await model.adecide({"question": "q"}, OPTIONS, "Pick one.")


def test_openrouter_close_closes_both_clients() -> None:
    model, client, async_client = make_model()
    async_client.aclose = AsyncMock()
    model.close()
    client.close.assert_called_once()
    async_client.aclose.assert_awaited_once()


def test_openrouter_context_manager() -> None:
    model, client, async_client = make_model()
    async_client.aclose = AsyncMock()
    with model as entered:
        assert entered is model
    client.close.assert_called_once()
    async_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_openrouter_async_context_manager_and_close_in_loop() -> None:
    model, client, async_client = make_model()
    async_client.aclose = AsyncMock()
    with pytest.raises(RuntimeError, match="aclose"):
        model.close()
    async with model as entered:
        assert entered is model
    client.close.assert_called_once()
    async_client.aclose.assert_awaited_once()


def test_decision_package_lazy_exports() -> None:
    assert "OpenRouterDecisionModel" in dir(decision_module)
    assert decision_module.OpenRouterDecisionModel is OpenRouterDecisionModel
    with pytest.raises(AttributeError):
        decision_module.NotAModel  # noqa: B018
