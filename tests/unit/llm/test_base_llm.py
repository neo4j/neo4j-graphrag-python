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
import warnings
from typing import Any, List, Optional, Type, Union

import pytest
from pydantic import BaseModel, ValidationError

from neo4j_graphrag.llm.base import BaseLLM, validate_invoke_input
from neo4j_graphrag.llm.types import LLMResponse, LLMUsage
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.rate_limit import NoOpRateLimitHandler

# ---------------------------------------------------------------------------
# LLMUsage
# ---------------------------------------------------------------------------


def test_llm_usage_defaults_to_none() -> None:
    usage = LLMUsage()
    assert usage.request_tokens is None
    assert usage.response_tokens is None
    assert usage.total_tokens is None


def test_llm_usage_accepts_explicit_values() -> None:
    usage = LLMUsage(request_tokens=10, response_tokens=20, total_tokens=30)
    assert usage.request_tokens == 10
    assert usage.response_tokens == 20
    assert usage.total_tokens == 30


def test_llm_usage_partial_values_keep_other_defaults() -> None:
    usage = LLMUsage(request_tokens=5)
    assert usage.request_tokens == 5
    assert usage.response_tokens is None
    assert usage.total_tokens is None


def test_llm_usage_rejects_non_integer_tokens() -> None:
    with pytest.raises(ValidationError):
        LLMUsage(request_tokens="bad")  # type: ignore[arg-type]


def test_llm_response_usage_is_none_by_default() -> None:
    response = LLMResponse(content="hello")
    assert response.usage is None


def test_llm_response_carries_usage() -> None:
    usage = LLMUsage(request_tokens=3, response_tokens=7, total_tokens=10)
    response = LLMResponse(content="hi", usage=usage)
    assert response.usage is not None
    assert response.usage.request_tokens == 3
    assert response.usage.response_tokens == 7
    assert response.usage.total_tokens == 10


# ---------------------------------------------------------------------------
# Minimal concrete subclass used across tests
# ---------------------------------------------------------------------------


class _ConcreteLLM(BaseLLM):
    """Minimal BaseLLM subclass for unit testing."""

    def invoke(
        self,
        input: List[LLMMessage],
        *,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=f"sync:{input[-1]['content']}")

    async def ainvoke(
        self,
        input: List[LLMMessage],
        *,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=f"async:{input[-1]['content']}")


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_llmbase_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BaseLLM(model_name="m")  # type: ignore[abstract]


def test_llmbase_sets_model_name() -> None:
    llm = _ConcreteLLM(model_name="my-model")
    assert llm.model_name == "my-model"


def test_llmbase_default_model_params_is_empty_dict() -> None:
    llm = _ConcreteLLM(model_name="m")
    assert llm.model_params == {}


def test_llmbase_accepts_model_params() -> None:
    llm = _ConcreteLLM(model_name="m", model_params={"temperature": 0.5})
    assert llm.model_params == {"temperature": 0.5}


def test_llmbase_accepts_custom_rate_limit_handler() -> None:
    handler = NoOpRateLimitHandler()
    llm = _ConcreteLLM(model_name="m", rate_limit_handler=handler)
    assert llm._rate_limit_handler is handler


def test_llmbase_init_does_not_emit_deprecation_warning() -> None:
    """BaseLLM.__init__ emits no deprecation warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ConcreteLLM(model_name="m")
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_warnings == []


# ---------------------------------------------------------------------------
# invoke / ainvoke
# ---------------------------------------------------------------------------


def test_invoke_accepts_message_list() -> None:
    llm = _ConcreteLLM(model_name="m")
    messages: List[LLMMessage] = [{"role": "user", "content": "hi"}]
    result = llm.invoke(messages)
    assert result.content == "sync:hi"


def test_invoke_accepts_response_format_kwarg() -> None:
    class MyModel(BaseModel):
        answer: str

    llm = _ConcreteLLM(model_name="m")
    messages: List[LLMMessage] = [{"role": "user", "content": "hi"}]
    # response_format must be keyword-only; this should not raise
    result = llm.invoke(messages, response_format=MyModel)
    assert result.content == "sync:hi"


@pytest.mark.asyncio
async def test_ainvoke_accepts_message_list() -> None:
    llm = _ConcreteLLM(model_name="m")
    messages: List[LLMMessage] = [{"role": "user", "content": "hi"}]
    result = await llm.ainvoke(messages)
    assert result.content == "async:hi"


# ---------------------------------------------------------------------------
# Tool calling defaults (inherited from BaseLLM)
# ---------------------------------------------------------------------------


def test_invoke_with_tools_raises_not_implemented() -> None:
    llm = _ConcreteLLM(model_name="m")
    with pytest.raises(NotImplementedError):
        llm.invoke_with_tools("hello", tools=[])


@pytest.mark.asyncio
async def test_ainvoke_with_tools_raises_not_implemented() -> None:
    llm = _ConcreteLLM(model_name="m")
    with pytest.raises(NotImplementedError):
        await llm.ainvoke_with_tools("hello", tools=[])


# ---------------------------------------------------------------------------
# validate_invoke_input
# ---------------------------------------------------------------------------


def test_validate_invoke_input_rejects_string() -> None:
    with pytest.raises(TypeError, match="list of LLMMessage"):
        validate_invoke_input("hello")


def test_validate_invoke_input_accepts_message_list() -> None:
    validate_invoke_input([{"role": "user", "content": "hi"}])
