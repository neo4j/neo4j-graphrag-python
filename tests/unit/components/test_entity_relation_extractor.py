from unittest.mock import MagicMock

import pytest
from neo4j_genai.components.entity_relation_extractor import (
    EntityRelationGraphModel,
    ERResultModel,
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_genai.exceptions import LLMGenerationError
from neo4j_genai.llm import LLMInterface, LLMResponse
from pydantic import ValidationError


@pytest.mark.asyncio
async def test_extractor_happy_path_empty_result() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(content='{"entities": [], "relations": []}')

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = ["some text"]
    result = await extractor.run(chunks=chunks)
    assert isinstance(result, ERResultModel)
    assert len(result.result) == len(chunks)
    chunk_result = result.result[0]
    assert chunk_result.error is False
    assert isinstance(chunk_result, EntityRelationGraphModel)
    assert chunk_result.entities == []
    assert chunk_result.relations == []


@pytest.mark.asyncio
async def test_extractor_happy_path_non_empty_result() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(
        content='{"entities": [{"id": "0", "label": "Person", "properties": {}}], "relations": []}'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = ["some text"]
    result = await extractor.run(chunks=chunks)
    assert isinstance(result, ERResultModel)
    assert len(result.result) == len(chunks)
    chunk_result = result.result[0]
    assert chunk_result.error is False
    assert isinstance(chunk_result, EntityRelationGraphModel)
    assert len(chunk_result.entities) == 1
    entity = chunk_result.entities[0]
    assert entity.id == "0"
    assert entity.label == "Person"
    assert entity.properties == {}
    assert chunk_result.relations == []


@pytest.mark.asyncio
async def test_extractor_missing_entity_id() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(
        content='{"entities": [{"label": "Person", "properties": {}}], "relations": []}'
    )
    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = ["some text"]
    with pytest.raises(ValidationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_invoke_failed() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.side_effect = LLMGenerationError()

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = ["some text"]
    with pytest.raises(LLMGenerationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_badly_formatted_json() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(
        content='{"entities": [{"id": "0", "label": "Person", "properties": {}}], "relations": [}'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = ["some text"]
    with pytest.raises(LLMGenerationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_invalid_json() -> None:
    """Test what happens when the returned JSON is valid JSON but
    does not match the expected Pydantic model"""
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(
        # missing "label" for entity
        content='{"entities": [{"id": 0, "entity_type": "Person", "properties": {}}], "relations": []}'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
    )
    chunks = ["some text"]
    with pytest.raises(ValidationError):
        await extractor.run(chunks=chunks)


@pytest.mark.asyncio
async def test_extractor_llm_badly_formatted_json_do_not_raise() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(
        content='{"entities": [{"id": "0", "label": "Person", "properties": {}}], "relations": [}'
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
        on_error=OnError.IGNORE,
    )
    chunks = ["some text"]
    res = await extractor.run(chunks=chunks)
    assert res.result[0].error is True


@pytest.mark.asyncio
async def test_extractor_custom_prompt() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(content='{"entities": [], "relations": []}')

    extractor = LLMEntityRelationExtractor(llm=llm, prompt_template="this is my prompt")
    chunks = ["some text"]
    await extractor.run(chunks=chunks)
    llm.invoke.assert_called_once_with("this is my prompt")
