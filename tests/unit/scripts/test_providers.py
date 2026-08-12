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
"""Unit tests for provider key validation.

The validators are the only place an API key is handled, so what they put in a
returned string matters: the installer prints those strings. Nothing here
performs real network I/O - urlopen is always replaced.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any, Iterator

import pytest
from examples_setup import providers

SECRET = "AIzaSy-super-secret-key-value"


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


@pytest.fixture(autouse=True)
def no_real_requests(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail loudly if a test reaches the network without saying so."""

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("test attempted real network I/O")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    yield


def test_unreachable_host_never_leaks_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installer prints this string verbatim.

    Regression: the Gemini key travelled in the query string and the error
    interpolated the whole URL, so any offline/DNS/proxy failure put the key in
    the user's scrollback.
    """
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float = 0) -> None:
        captured["request"] = request
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ok, detail = providers._validate_gemini(SECRET)

    assert ok is False
    assert SECRET not in detail
    assert "generativelanguage.googleapis.com" in detail


def test_gemini_sends_the_key_as_a_header_not_in_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key in a query string is logged by every proxy along the way."""
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float = 0) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["headers"] = request.headers
        return _FakeResponse(200)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ok, _ = providers._validate_gemini(SECRET)

    assert ok is True
    assert SECRET not in captured["url"]
    assert SECRET in captured["headers"].values()


@pytest.mark.parametrize(
    "status,expected_ok",
    [
        (401, False),
        (403, False),
        (429, True),  # rate limited still proves the key authenticates
        (500, False),
    ],
)
def test_http_error_codes(
    monkeypatch: pytest.MonkeyPatch, status: int, expected_ok: bool
) -> None:
    def fake_urlopen(request: Any, timeout: float = 0) -> None:
        raise urllib.error.HTTPError("url", status, "err", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    ok, detail = providers._validate_gemini(SECRET)

    assert ok is expected_ok
    assert SECRET not in detail


def test_every_validator_keeps_the_key_out_of_its_detail_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever a provider's validator returns, it gets printed."""

    def fake_urlopen(request: Any, timeout: float = 0) -> None:
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    validators = [
        provider.validator
        for provider in providers.PROVIDERS.values()
        if provider.validator is not None
    ]
    assert validators, "expected at least one provider with a validator"
    for validator in validators:
        _, detail = validator(SECRET)
        assert SECRET not in detail
