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
from unittest.mock import MagicMock, patch

import neo4j
import pytest
from neo4j_graphrag.experimental.components.entity_relation_extractor import OnError
from neo4j_graphrag.experimental.components.schema import SchemaEntity, SchemaRelation
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm.base import LLMInterface
from pydantic import ValidationError


@pytest.mark.asyncio
async def test_knowledge_graph_builder_init_with_text() -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        from_pdf=False,
    )

    assert kg_builder.llm == llm
    assert kg_builder.driver == driver
    assert kg_builder.from_pdf is False
    assert kg_builder.entities == []
    assert kg_builder.relations == []
    assert kg_builder.potential_schema == []

    text_input = "May thy knife chip and shatter."

    with patch.object(
        kg_builder.pipeline,
        "run",
        return_value=PipelineResult(run_id="test_run", result=None),
    ) as mock_run:
        await kg_builder.run_async(text=text_input)
        mock_run.assert_called_once()
        pipe_inputs = mock_run.call_args[0][0]
        assert pipe_inputs["splitter"]["text"] == text_input


@pytest.mark.asyncio
async def test_knowledge_graph_builder_init_with_file_path() -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        from_pdf=True,
    )

    assert kg_builder.llm == llm
    assert kg_builder.driver == driver
    assert kg_builder.from_pdf is True
    assert kg_builder.entities == []
    assert kg_builder.relations == []
    assert kg_builder.potential_schema == []

    file_path = "path/to/test.pdf"

    with patch.object(
        kg_builder.pipeline,
        "run",
        return_value=PipelineResult(run_id="test_run", result=None),
    ) as mock_run:
        await kg_builder.run_async(file_path=file_path)
        mock_run.assert_called_once()
        pipe_inputs = mock_run.call_args[0][0]
        assert pipe_inputs["pdf_loader"]["filepath"] == file_path


@pytest.mark.asyncio
async def test_knowledge_graph_builder_run_with_both_inputs() -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        from_pdf=True,
    )

    text_input = "May thy knife chip and shatter."
    file_path = "path/to/test.pdf"

    with pytest.raises(PipelineDefinitionError) as exc_info:
        await kg_builder.run_async(file_path=file_path, text=text_input)

    assert "Expected 'file_path' argument when 'from_pdf' is True." in str(
        exc_info.value
    ) or "Expected 'text' argument when 'from_pdf' is False." in str(exc_info.value)


@pytest.mark.asyncio
async def test_knowledge_graph_builder_run_with_no_inputs() -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        from_pdf=True,  # or False
    )

    with pytest.raises(PipelineDefinitionError) as exc_info:
        await kg_builder.run_async()

    assert "Expected 'file_path' argument when 'from_pdf' is True." in str(
        exc_info.value
    ) or "Expected 'text' argument when 'from_pdf' is False." in str(exc_info.value)


@pytest.mark.asyncio
async def test_knowledge_graph_builder_document_info_with_file() -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        from_pdf=True,
    )

    file_path = "path/to/test.pdf"

    with patch.object(
        kg_builder.pipeline,
        "run",
        return_value=PipelineResult(run_id="test_run", result=None),
    ) as mock_run:
        await kg_builder.run_async(file_path=file_path)

        pipe_inputs = mock_run.call_args[0][0]
        assert "pdf_loader" in pipe_inputs
        assert pipe_inputs["pdf_loader"] == {"filepath": file_path}
        assert "extractor" not in pipe_inputs


@pytest.mark.asyncio
async def test_knowledge_graph_builder_document_info_with_text() -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        from_pdf=False,
    )

    text_input = "May thy knife chip and shatter."

    with patch.object(
        kg_builder.pipeline,
        "run",
        return_value=PipelineResult(run_id="test_run", result=None),
    ) as mock_run:
        await kg_builder.run_async(text=text_input)

        pipe_inputs = mock_run.call_args[0][0]
        assert "splitter" in pipe_inputs
        assert pipe_inputs["splitter"] == {"text": text_input}


@pytest.mark.asyncio
async def test_knowledge_graph_builder_with_entities_and_file() -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)

    entities = ["Document", "Section"]
    relations = ["CONTAINS"]
    potential_schema = [("Document", "CONTAINS", "Section")]

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        entities=entities,
        relations=relations,
        potential_schema=potential_schema,
        from_pdf=True,
    )

    internal_entities = [SchemaEntity(label=label) for label in entities]
    internal_relations = [SchemaRelation(label=label) for label in relations]
    assert kg_builder.entities == internal_entities
    assert kg_builder.relations == internal_relations
    assert kg_builder.potential_schema == potential_schema

    file_path = "path/to/test.pdf"

    with patch.object(
        kg_builder.pipeline,
        "run",
        return_value=PipelineResult(run_id="test_run", result=None),
    ) as mock_run:
        await kg_builder.run_async(file_path=file_path)
        pipe_inputs = mock_run.call_args[0][0]
        assert pipe_inputs["schema"]["entities"] == internal_entities
        assert pipe_inputs["schema"]["relations"] == internal_relations
        assert pipe_inputs["schema"]["potential_schema"] == potential_schema


def test_simple_kg_pipeline_on_error_conversion() -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        on_error="RAISE",
    )

    assert kg_builder.on_error == OnError.RAISE


def test_simple_kg_pipeline_on_error_invalid_value() -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)

    with pytest.raises(ValidationError) as exc_info:
        SimpleKGPipeline(
            llm=llm,
            driver=driver,
            on_error="IGNORE",
        )

    assert "Input should be 'RAISE' or 'CONTINUE'" in str(exc_info.value)
