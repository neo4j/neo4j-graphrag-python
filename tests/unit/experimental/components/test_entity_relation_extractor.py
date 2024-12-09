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

import json
from unittest.mock import MagicMock, patch

import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
    balance_curly_braces,
    fix_invalid_json,
)
from neo4j_graphrag.experimental.components.pdf_loader import DocumentInfo
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.experimental.pipeline.exceptions import JSONRepairError
from neo4j_graphrag.llm import LLMInterface, LLMResponse


@pytest.mark.asyncio
async def test_extractor_happy_path_no_entities_no_document() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(content='{"nodes": [], "relationships": []}')

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text", index=0)])
    result = await extractor.run(chunks=chunks)
    assert isinstance(result, Neo4jGraph)
    # only one Chunk node (no document info provided)
    assert len(result.nodes) == 1
    assert result.nodes[0].label == "Chunk"
    assert result.relationships == []


@pytest.mark.asyncio
async def test_extractor_happy_path_no_entities() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(content='{"nodes": [], "relationships": []}')

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text", index=0)])
    document_info = DocumentInfo(path="path")
    result = await extractor.run(chunks=chunks, document_info=document_info)
    assert isinstance(result, Neo4jGraph)
    # one Chunk node and one Document node
    assert len(result.nodes) == 2
    assert set(n.label for n in result.nodes) == {"Chunk", "Document"}
    assert len(result.relationships) == 1
    assert result.relationships[0].type == "FROM_DOCUMENT"


@pytest.mark.asyncio
async def test_extractor_happy_path_no_entities_no_lexical_graph() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(content='{"nodes": [], "relationships": []}')

    extractor = LLMEntityRelationExtractor(
        llm=llm,
        create_lexical_graph=False,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text", index=0)])
    document_info = DocumentInfo(path="path")
    graph = await extractor.run(chunks=chunks, document_info=document_info)
    assert graph.nodes == []
    assert graph.relationships == []


@pytest.mark.asyncio
async def test_extractor_happy_path_non_empty_result() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": {}}], "relationships": []}'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text", index=0)])
    document_info = DocumentInfo(path="path")
    result = await extractor.run(chunks=chunks, document_info=document_info)
    assert isinstance(result, Neo4jGraph)
    assert len(result.nodes) == 3
    doc = result.nodes[0]
    assert doc.label == "Document"
    chunk_entity = result.nodes[1]
    assert chunk_entity.label == "Chunk"
    entity = result.nodes[2]
    assert entity.id.endswith("0:0")
    assert entity.label == "Person"
    assert entity.properties == {"chunk_index": 0}
    assert len(result.relationships) == 2
    assert result.relationships[0].type == "FROM_DOCUMENT"
    assert result.relationships[0].start_node_id.endswith(":0")
    assert result.relationships[0].end_node_id == "path"
    assert result.relationships[1].type == "FROM_CHUNK"


@pytest.mark.asyncio
async def test_extractor_missing_entity_id() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"label": "Person", "properties": {}}], "relationships": []}'
    )
    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text", index=0)])
    with pytest.raises(LLMGenerationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_ainvoke_failed() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.side_effect = LLMGenerationError()

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text", index=0)])
    with pytest.raises(LLMGenerationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_unfixable_json() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": {}}], "relationships": }'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text", index=0)])

    with pytest.raises(LLMGenerationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_invalid_json() -> None:
    """Test what happens when the returned JSON is valid JSON but
    does not match the expected Pydantic model"""
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        # missing "label" for entity
        content='{"nodes": [{"id": 0, "entity_type": "Person", "properties": {}}], "relationships": []}'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text", index=0)])
    with pytest.raises(LLMGenerationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_badly_formatted_json_gets_fixed() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": {}}], "relationships": [}'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
        on_error=OnError.IGNORE,
        create_lexical_graph=False,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text", index=0)])
    res = await extractor.run(chunks=chunks)

    assert len(res.nodes) == 1
    assert res.nodes[0].label == "Person"
    assert res.nodes[0].properties == {"chunk_index": 0}
    assert res.nodes[0].embedding_properties is None
    assert res.relationships == []


@pytest.mark.asyncio
async def test_extractor_custom_prompt() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(content='{"nodes": [], "relationships": []}')

    extractor = LLMEntityRelationExtractor(llm=llm, prompt_template="this is my prompt")
    chunks = TextChunks(chunks=[TextChunk(text="some text", index=0)])
    await extractor.run(chunks=chunks)
    llm.ainvoke.assert_called_once_with("this is my prompt")


def test_fix_invalid_json_empty_result() -> None:
    json_string = "invalid json"

    with patch("json_repair.repair_json", return_value=""):
        with pytest.raises(JSONRepairError):
            fix_invalid_json(json_string)


def test_fix_unquoted_keys() -> None:
    json_string = '{name: "John", age: "30"}'
    expected_result = '{"name": "John", "age": "30"}'

    fixed_json = fix_invalid_json(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_fix_unquoted_string_values() -> None:
    json_string = '{"name": John, "age": 30}'
    expected_result = '{"name": "John", "age": 30}'

    fixed_json = fix_invalid_json(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_remove_trailing_commas() -> None:
    json_string = '{"name": "John", "age": 30,}'
    expected_result = '{"name": "John", "age": 30}'

    fixed_json = fix_invalid_json(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_fix_excessive_braces() -> None:
    json_string = '{{"name": "John"}}'
    expected_result = '{"name": "John"}'

    fixed_json = fix_invalid_json(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_fix_multiple_issues() -> None:
    json_string = '{name: John, "hobbies": ["reading", "swimming",], "age": 30}'
    expected_result = '{"name": "John", "hobbies": ["reading", "swimming"], "age": 30}'

    fixed_json = fix_invalid_json(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_fix_null_values() -> None:
    json_string = '{"name": John, "nickname": null}'
    expected_result = '{"name": "John", "nickname": null}'

    fixed_json = fix_invalid_json(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_fix_numeric_values() -> None:
    json_string = '{"age": 30, "score": 95.5}'
    expected_result = '{"age": 30, "score": 95.5}'

    fixed_json = fix_invalid_json(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_balance_curly_braces_missing_closing() -> None:
    json_string = '{"name": "John", "hobbies": {"reading": "yes"'
    expected_result = '{"name": "John", "hobbies": {"reading": "yes"}}'

    fixed_json = balance_curly_braces(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_balance_curly_braces_extra_closing() -> None:
    json_string = '{"name": "John", "hobbies": {"reading": "yes"}}}'
    expected_result = '{"name": "John", "hobbies": {"reading": "yes"}}'

    fixed_json = balance_curly_braces(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_balance_curly_braces_balanced_input() -> None:
    json_string = '{"name": "John", "hobbies": {"reading": "yes"}, "age": 30}'
    expected_result = json_string

    fixed_json = balance_curly_braces(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_balance_curly_braces_nested_structure() -> None:
    json_string = '{"person": {"name": "John", "hobbies": {"reading": "yes"}}}'
    expected_result = json_string

    fixed_json = balance_curly_braces(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_balance_curly_braces_unbalanced_nested() -> None:
    json_string = '{"person": {"name": "John", "hobbies": {"reading": "yes"}}'
    expected_result = '{"person": {"name": "John", "hobbies": {"reading": "yes"}}}'

    fixed_json = balance_curly_braces(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_balance_curly_braces_unmatched_openings() -> None:
    json_string = '{"name": "John", "hobbies": {"reading": "yes"'
    expected_result = '{"name": "John", "hobbies": {"reading": "yes"}}'

    fixed_json = balance_curly_braces(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_balance_curly_braces_unmatched_closings() -> None:
    json_string = '{"name": "John", "hobbies": {"reading": "yes"}}}'
    expected_result = '{"name": "John", "hobbies": {"reading": "yes"}}'

    fixed_json = balance_curly_braces(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_balance_curly_braces_complex_structure() -> None:
    json_string = (
        '{"name": "John", "details": {"age": 30, "hobbies": {"reading": "yes"}}}'
    )
    expected_result = json_string

    fixed_json = balance_curly_braces(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_balance_curly_braces_incorrect_nested_closings() -> None:
    json_string = '{"key1": {"key2": {"reading": "yes"}}, "key3": {"age": 30}}}'
    expected_result = '{"key1": {"key2": {"reading": "yes"}}, "key3": {"age": 30}}'

    fixed_json = balance_curly_braces(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_balance_curly_braces_braces_inside_string() -> None:
    json_string = '{"name": "John", "example": "a{b}c", "age": 30}'
    expected_result = json_string

    fixed_json = balance_curly_braces(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result


def test_balance_curly_braces_unbalanced_with_string() -> None:
    json_string = '{"name": "John", "example": "a{b}c", "hobbies": {"reading": "yes"'
    expected_result = (
        '{"name": "John", "example": "a{b}c", "hobbies": {"reading": "yes"}}'
    )

    fixed_json = balance_curly_braces(json_string)

    assert json.loads(fixed_json)
    assert fixed_json == expected_result
