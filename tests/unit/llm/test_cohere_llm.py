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
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import cohere.core
import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.cohere_llm import CohereLLM


@patch("neo4j_graphrag.llm.cohere.cohere", None)
def test_cohere_llm_missing_dependency() -> None:
    with pytest.raises(ImportError):
        CohereLLM(model_name="something")


@patch("neo4j_graphrag.llm.cohere.cohere")
def test_cohere_embedder_happy_path(mock_cohere: Mock) -> None:
    mock_cohere.Client.return_value.chat.return_value = MagicMock(
        text="cohere response text"
    )
    embedder = CohereLLM(model_name="something")
    res = embedder.invoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "cohere response text"


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.cohere.cohere")
async def test_cohere_embedder_happy_path_async(mock_cohere: Mock) -> None:
    async_mock = Mock()
    async_mock.chat = AsyncMock(return_value=MagicMock(text="cohere response text"))
    mock_cohere.AsyncClient.return_value = async_mock
    embedder = CohereLLM(model_name="something")
    res = await embedder.ainvoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "cohere response text"


@patch("neo4j_graphrag.llm.cohere.cohere")
def test_cohere_embedder_failed(mock_cohere: Mock) -> None:
    mock_cohere.Client.return_value.chat.side_effect = cohere.core.ApiError
    embedder = CohereLLM(model_name="something")
    with pytest.raises(LLMGenerationError) as excinfo:
        embedder.invoke("my text")
    assert "ApiError" in str(excinfo)


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.cohere.cohere")
async def test_cohere_embedder_failed_async(mock_cohere: Mock) -> None:
    mock_cohere.AsyncClient.return_value.chat.side_effect = cohere.core.ApiError
    embedder = CohereLLM(model_name="something")
    with pytest.raises(LLMGenerationError) as excinfo:
        await embedder.ainvoke("my text")
    assert "ApiError" in str(excinfo)
