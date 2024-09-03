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
from unittest.mock import MagicMock

import pytest
from neo4j_genai.exceptions import LLMGenerationError
from neo4j_genai.experimental.components.entity_relation_extractor import (
    EntityRelationExtractor,
    LLMEntityRelationExtractor,
    OnError,
    balance_curly_braces,
    fix_invalid_json,
)
from neo4j_genai.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    TextChunk,
    TextChunks,
)
from neo4j_genai.llm import LLMInterface, LLMResponse


def test_create_chunk_node_no_metadata() -> None:
    # instantiating an abstract class to test common methods
    extractor = EntityRelationExtractor()  # type: ignore
    node = extractor.create_chunk_node(
        chunk=TextChunk(text="text chunk"), chunk_id="10"
    )
    assert isinstance(node, Neo4jNode)
    assert node.id == "10"
    assert node.properties == {"text": "text chunk"}
    assert node.embedding_properties == {}


def test_create_chunk_node_metadata_no_embedding() -> None:
    # instantiating an abstract class to test common methods
    extractor = EntityRelationExtractor()  # type: ignore
    node = extractor.create_chunk_node(
        chunk=TextChunk(text="text chunk", metadata={"status": "ok"}), chunk_id="10"
    )
    assert isinstance(node, Neo4jNode)
    assert node.id == "10"
    assert node.properties == {"text": "text chunk", "status": "ok"}
    assert node.embedding_properties == {}


def test_create_chunk_node_metadata_embedding() -> None:
    # instantiating an abstract class to test common methods
    extractor = EntityRelationExtractor()  # type: ignore
    node = extractor.create_chunk_node(
        chunk=TextChunk(
            text="text chunk", metadata={"status": "ok", "embedding": [1, 2, 3]}
        ),
        chunk_id="10",
    )
    assert isinstance(node, Neo4jNode)
    assert node.id == "10"
    assert node.properties == {"text": "text chunk", "status": "ok"}
    assert node.embedding_properties == {"embedding": [1, 2, 3]}


@pytest.mark.asyncio
async def test_extractor_happy_path_no_entities() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(content='{"nodes": [], "relationships": []}')

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    result = await extractor.run(chunks=chunks)
    assert isinstance(result, Neo4jGraph)
    # only one Chunk node
    assert len(result.nodes) == 1
    assert result.nodes[0].label == "Chunk"
    assert result.relationships == []


@pytest.mark.asyncio
async def test_extractor_happy_path_no_entities_no_lexical_graph() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(content='{"nodes": [], "relationships": []}')

    extractor = LLMEntityRelationExtractor(
        llm=llm,
        create_lexical_graph=False,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    result = await extractor.run(chunks=chunks)
    assert result.nodes == []
    assert result.relationships == []


@pytest.mark.asyncio
async def test_extractor_happy_path_non_empty_result() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": {}}], "relationships": []}'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    result = await extractor.run(chunks=chunks)
    assert isinstance(result, Neo4jGraph)
    assert len(result.nodes) == 2
    entity = result.nodes[0]
    assert entity.id.endswith("0:0")
    assert entity.label == "Person"
    assert entity.properties == {"chunk_index": 0}
    chunk_entity = result.nodes[1]
    assert chunk_entity.label == "Chunk"
    assert len(result.relationships) == 1
    assert result.relationships[0].type == "FROM_CHUNK"


@pytest.mark.asyncio
async def test_extractor_missing_entity_id() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"label": "Person", "properties": {}}], "relationships": []}'
    )
    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    with pytest.raises(LLMGenerationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_ainvoke_failed() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.side_effect = LLMGenerationError()

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    with pytest.raises(LLMGenerationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_badly_formatted_json() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": {}}], "relationships": [}'
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
    llm.ainvoke.return_value = LLMResponse(
        # missing "label" for entity
        content='{"nodes": [{"id": 0, "entity_type": "Person", "properties": {}}], "relationships": []}'
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
    llm.ainvoke.return_value = LLMResponse(
        content='{"nodes": [{"id": "0", "label": "Person", "properties": {}}], "relationships": [}'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
        on_error=OnError.IGNORE,
        create_lexical_graph=False,
    )
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    res = await extractor.run(chunks=chunks)
    assert res.nodes == []
    assert res.relationships == []


@pytest.mark.asyncio
async def test_extractor_custom_prompt() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.ainvoke.return_value = LLMResponse(content='{"nodes": [], "relationships": []}')

    extractor = LLMEntityRelationExtractor(llm=llm, prompt_template="this is my prompt")
    chunks = TextChunks(chunks=[TextChunk(text="some text")])
    await extractor.run(chunks=chunks)
    llm.ainvoke.assert_called_once_with("this is my prompt")


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
