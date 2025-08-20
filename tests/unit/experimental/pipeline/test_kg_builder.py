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
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import neo4j
import pytest
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.components.types import (
    LexicalGraphConfig,
)
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm.base import LLMInterface


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 23, 0), False, False),
)
@pytest.mark.asyncio
async def test_knowledge_graph_builder_document_info_with_file(_: Mock) -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)
    embedder = MagicMock(spec=Embedder)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        from_pdf=True,
    )

    file_path = "path/to/test.pdf"

    with patch.object(
        kg_builder.runner.pipeline,
        "run",
        return_value=PipelineResult(run_id="test_run", result=None),
    ) as mock_run:
        await kg_builder.run_async(
            file_path=file_path,
            document_metadata={"source": "google drive"}
        )

        pipe_inputs = mock_run.call_args[1]["data"]
        assert "pdf_loader" in pipe_inputs
        assert pipe_inputs["pdf_loader"] == {"filepath": file_path, "metadata": {"source": "google drive"}}
        assert "extractor" not in pipe_inputs


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 23, 0), False, False),
)
@pytest.mark.asyncio
async def test_knowledge_graph_builder_document_info_with_text(_: Mock) -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)
    embedder = MagicMock(spec=Embedder)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        from_pdf=False,
    )

    text_input = "May thy knife chip and shatter."

    with patch.object(
        kg_builder.runner.pipeline,
        "run",
        return_value=PipelineResult(run_id="test_run", result=None),
    ) as mock_run:
        await kg_builder.run_async(
            text=text_input,
            document_path="my_document.txt",
            document_metadata={"source": "google drive"},
        )

        pipe_inputs = mock_run.call_args[1]["data"]
        assert "splitter" in pipe_inputs
        assert pipe_inputs["splitter"] == {"text": text_input}
        assert pipe_inputs["extractor"]["document_info"]["path"] == "my_document.txt"
        assert pipe_inputs["extractor"]["document_info"]["metadata"] == {
            "source": "google drive"
        }


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 23, 0), False, False),
)
@pytest.mark.asyncio
async def test_knowledge_graph_builder_with_entities_and_file(_: Mock) -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)
    embedder = MagicMock(spec=Embedder)

    entities = ["Document", "Section"]
    relations = ["CONTAINS"]
    potential_schema = [("Document", "CONTAINS", "Section")]

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        entities=entities,
        relations=relations,
        potential_schema=potential_schema,
        from_pdf=True,
    )

    file_path = "path/to/test.pdf"

    with patch.object(
        kg_builder.runner.pipeline,
        "run",
        return_value=PipelineResult(run_id="test_run", result=None),
    ) as mock_run:
        await kg_builder.run_async(file_path=file_path)
        pipe_inputs = mock_run.call_args[1]["data"]
        assert pipe_inputs["schema"]["node_types"] == entities
        assert pipe_inputs["schema"]["relationship_types"] == relations
        assert pipe_inputs["schema"]["patterns"] == potential_schema


def test_simple_kg_pipeline_on_error_invalid_value() -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)
    embedder = MagicMock(spec=Embedder)

    with pytest.raises(PipelineDefinitionError):
        SimpleKGPipeline(
            llm=llm,
            driver=driver,
            embedder=embedder,
            on_error="INVALID_VALUE",
        )


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 23, 0), False, False),
)
@pytest.mark.asyncio
async def test_knowledge_graph_builder_with_lexical_graph_config(_: Mock) -> None:
    llm = MagicMock(spec=LLMInterface)
    driver = MagicMock(spec=neo4j.Driver)
    embedder = MagicMock(spec=Embedder)

    chunk_node_label = "TestChunk"
    document_nodel_label = "TestDocument"
    lexical_graph_config = LexicalGraphConfig(
        chunk_node_label=chunk_node_label, document_node_label=document_nodel_label
    )

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        from_pdf=False,
        lexical_graph_config=lexical_graph_config,
    )

    text_input = "May thy knife chip and shatter."

    with patch.object(
        kg_builder.runner.pipeline,
        "run",
        return_value=PipelineResult(run_id="test_run", result=None),
    ) as mock_run:
        await kg_builder.run_async(text=text_input)

        pipe_inputs = mock_run.call_args[1]["data"]
        assert "extractor" in pipe_inputs
        assert pipe_inputs["extractor"]["lexical_graph_config"] == lexical_graph_config
        assert pipe_inputs["extractor"]["document_info"] is not None
        assert pipe_inputs["extractor"]["document_info"]["path"] == "document.txt"
