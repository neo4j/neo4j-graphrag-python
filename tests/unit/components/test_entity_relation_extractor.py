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

from unittest.mock import MagicMock

import pytest
from neo4j_genai.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_genai.components.types import Neo4jGraph, TextChunk, TextChunks
from neo4j_genai.exceptions import LLMGenerationError
from neo4j_genai.llm import LLMInterface, LLMResponse


@pytest.mark.asyncio
async def test_extractor_happy_path_empty_result() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(content='{"nodes": [], "relationships": []}')

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    result = await extractor.run(chunks=chunks)
    assert isinstance(result, Neo4jGraph)
    assert result.nodes == []
    assert result.relationships == []


@pytest.mark.asyncio
async def test_extractor_happy_path_non_empty_result() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": []}], "relationships": []}'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    result = await extractor.run(chunks=chunks)
    assert isinstance(result, Neo4jGraph)
    entity = result.nodes[0]
    assert entity.id == "0:0"
    assert entity.label == "Person"
    assert entity.properties == []
    assert result.relationships == []


@pytest.mark.asyncio
async def test_extractor_missing_entity_id() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(
        content='{"nodes": [{"label": "Person", "properties": []}], "relationships": []}'
    )
    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    with pytest.raises(LLMGenerationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_invoke_failed() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.side_effect = LLMGenerationError()

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    with pytest.raises(LLMGenerationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_badly_formatted_json() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": []}], "relationships": [}'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    with pytest.raises(LLMGenerationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_invalid_json() -> None:
    """Test what happens when the returned JSON is valid JSON but
    does not match the expected Pydantic model"""
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(
        # missing "label" for entity
        content='{"nodes": [{"id": 0, "entity_type": "Person", "properties": []}], "relationships": []}'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    with pytest.raises(LLMGenerationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_badly_formatted_json_do_not_raise() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": []}], "relationships": [}'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
        on_error=OnError.IGNORE,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    res = await extractor.run(chunks=chunks)
    assert res.nodes == []
    assert res.relationships == []


@pytest.mark.asyncio
async def test_extractor_custom_prompt() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(content='{"nodes": [], "relationships": []}')

    extractor = LLMEntityRelationExtractor(llm=llm, prompt_template="this is my prompt")
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    await extractor.run(chunks=chunks)
    llm.invoke.assert_called_once_with("this is my prompt")
