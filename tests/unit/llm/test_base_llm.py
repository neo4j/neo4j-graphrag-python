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
from pydantic import BaseModel

from neo4j_graphrag.llm.base import LLMBase
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.rate_limit import NoOpRateLimitHandler


# ---------------------------------------------------------------------------
# Minimal concrete subclass used across tests
# ---------------------------------------------------------------------------

class _ConcreteLLM(LLMBase):
    """Minimal LLMBase subclass for unit testing."""

    def invoke(
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return LLMResponse(content=f"v1:{input}")
        return LLMResponse(content="v2:list")

    async def ainvoke(
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(input, str):
            return LLMResponse(content=f"async_v1:{input}")
        return LLMResponse(content="async_v2:list")


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_llmbase_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        LLMBase(model_name="m")  # type: ignore[abstract]


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
    """LLMBase.__init__ delegates to LLMInterfaceV2, which has no deprecation warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ConcreteLLM(model_name="m")
    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation_warnings == []


# ---------------------------------------------------------------------------
# invoke routing
# ---------------------------------------------------------------------------

def test_invoke_with_str_routes_to_v1() -> None:
    llm = _ConcreteLLM(model_name="m")
    result = llm.invoke("hello")
    assert result.content == "v1:hello"


def test_invoke_with_list_routes_to_v2() -> None:
    llm = _ConcreteLLM(model_name="m")
    messages: List[LLMMessage] = [{"role": "user", "content": "hi"}]
    result = llm.invoke(messages)
    assert result.content == "v2:list"


def test_invoke_v2_accepts_response_format_kwarg() -> None:
    class MyModel(BaseModel):
        answer: str

    llm = _ConcreteLLM(model_name="m")
    messages: List[LLMMessage] = [{"role": "user", "content": "hi"}]
    # response_format must be keyword-only; this should not raise
    result = llm.invoke(messages, response_format=MyModel)
    assert result.content == "v2:list"


# ---------------------------------------------------------------------------
# ainvoke routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ainvoke_with_str_routes_to_v1() -> None:
    llm = _ConcreteLLM(model_name="m")
    result = await llm.ainvoke("hello")
    assert result.content == "async_v1:hello"


@pytest.mark.asyncio
async def test_ainvoke_with_list_routes_to_v2() -> None:
    llm = _ConcreteLLM(model_name="m")
    messages: List[LLMMessage] = [{"role": "user", "content": "hi"}]
    result = await llm.ainvoke(messages)
    assert result.content == "async_v2:list"


# ---------------------------------------------------------------------------
# Tool calling defaults (inherited from LLMInterface)
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
