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

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from neo4j_graphrag.llm.vertexai_llm import VertexAILLM

pytestmark = pytest.mark.google


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel", None)
def test_vertexai_llm_missing_dependency() -> None:
    with pytest.raises(ImportError):
        VertexAILLM(model_name="gemini-1.5-flash-001")


@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
def test_vertexai_invoke_happy_path(GenerativeModelMock: MagicMock) -> None:
    mock_response = Mock()
    mock_response.text = "Return text"
    mock_model = GenerativeModelMock.return_value
    mock_model.generate_content.return_value = mock_response
    model_params = {"temperature": 0.5}
    llm = VertexAILLM("gemini-1.5-flash-001", model_params)
    input_text = "may thy knife chip and shatter"
    response = llm.invoke(input_text)
    assert response.content == "Return text"
    llm.model.generate_content.assert_called_once_with(input_text, **model_params)


@pytest.mark.asyncio
@patch("neo4j_graphrag.llm.vertexai_llm.GenerativeModel")
async def test_vertexai_ainvoke_happy_path(GenerativeModelMock: MagicMock) -> None:
    mock_response = AsyncMock()
    mock_response.text = "Return text"
    mock_model = GenerativeModelMock.return_value
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    model_params = {"temperature": 0.5}
    llm = VertexAILLM("gemini-1.5-flash-001", model_params)
    input_text = "may thy knife chip and shatter"
    response = await llm.ainvoke(input_text)
    assert response.content == "Return text"
    llm.model.generate_content_async.assert_called_once_with(input_text, **model_params)
