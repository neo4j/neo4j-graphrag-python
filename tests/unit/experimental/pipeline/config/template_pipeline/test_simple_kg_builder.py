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
from unittest.mock import Mock, patch

import neo4j
import pytest
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaRelation,
)
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.config.object_config import ComponentConfig
from neo4j_graphrag.experimental.pipeline.config.template_pipeline import (
    SimpleKGPipelineConfig,
)
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm import LLMInterface


def test_simple_kg_pipeline_config_pdf_loader_from_pdf_is_false() -> None:
    config = SimpleKGPipelineConfig(from_pdf=False)
    assert config._get_pdf_loader() is None


def test_simple_kg_pipeline_config_pdf_loader_from_pdf_is_true() -> None:
    config = SimpleKGPipelineConfig(from_pdf=True)
    assert isinstance(config._get_pdf_loader(), PdfLoader)


def test_simple_kg_pipeline_config_pdf_loader_from_pdf_is_true_class_overwrite() -> (
    None
):
    my_pdf_loader = PdfLoader()
    config = SimpleKGPipelineConfig(from_pdf=True, pdf_loader=my_pdf_loader)  # type: ignore
    assert config._get_pdf_loader() == my_pdf_loader


def test_simple_kg_pipeline_config_pdf_loader_class_overwrite_but_from_pdf_is_false() -> (
    None
):
    my_pdf_loader = PdfLoader()
    config = SimpleKGPipelineConfig(from_pdf=False, pdf_loader=my_pdf_loader)  # type: ignore
    assert config._get_pdf_loader() is None


@patch("neo4j_graphrag.experimental.pipeline.config.object_config.ComponentType.parse")
def test_simple_kg_pipeline_config_pdf_loader_from_pdf_is_true_class_overwrite_from_config(
    mock_component_parse: Mock,
) -> None:
    my_pdf_loader_config = ComponentConfig(
        class_="",
    )
    my_pdf_loader = PdfLoader()
    mock_component_parse.return_value = my_pdf_loader
    config = SimpleKGPipelineConfig(
        from_pdf=True,
        pdf_loader=my_pdf_loader_config,  # type: ignore
    )
    assert config._get_pdf_loader() == my_pdf_loader


def test_simple_kg_pipeline_config_text_splitter() -> None:
    config = SimpleKGPipelineConfig()
    assert isinstance(config._get_splitter(), FixedSizeSplitter)


@patch("neo4j_graphrag.experimental.pipeline.config.object_config.ComponentType.parse")
def test_simple_kg_pipeline_config_text_splitter_overwrite(
    mock_component_parse: Mock,
) -> None:
    my_text_splitter_config = ComponentConfig(
        class_="",
    )
    my_text_splitter = FixedSizeSplitter()
    mock_component_parse.return_value = my_text_splitter
    config = SimpleKGPipelineConfig(
        text_splitter=my_text_splitter_config,  # type: ignore
    )
    assert config._get_splitter() == my_text_splitter


@patch(
    "neo4j_graphrag.experimental.pipeline.config.template_pipeline.simple_kg_builder.SimpleKGPipelineConfig.get_default_embedder"
)
def test_simple_kg_pipeline_config_chunk_embedder(
    mock_embedder: Mock, embedder: Embedder
) -> None:
    mock_embedder.return_value = embedder
    config = SimpleKGPipelineConfig()
    chunk_embedder = config._get_chunk_embedder()
    assert isinstance(chunk_embedder, TextChunkEmbedder)
    assert chunk_embedder._embedder == embedder


def test_simple_kg_pipeline_config_schema() -> None:
    config = SimpleKGPipelineConfig()
    assert isinstance(config._get_schema(), SchemaBuilder)


def test_simple_kg_pipeline_config_schema_run_params() -> None:
    config = SimpleKGPipelineConfig(
        entities=["Person"],
        relations=["KNOWS"],
        potential_schema=[("Person", "KNOWS", "Person")],
    )
    assert config._get_run_params_for_schema() == {
        "entities": [SchemaEntity(label="Person")],
        "relations": [SchemaRelation(label="KNOWS")],
        "potential_schema": [
            ("Person", "KNOWS", "Person"),
        ],
    }


@patch(
    "neo4j_graphrag.experimental.pipeline.config.template_pipeline.simple_kg_builder.SimpleKGPipelineConfig.get_default_llm"
)
def test_simple_kg_pipeline_config_extractor(mock_llm: Mock, llm: LLMInterface) -> None:
    mock_llm.return_value = llm
    config = SimpleKGPipelineConfig(
        on_error="IGNORE",  # type: ignore
        prompt_template=ERExtractionTemplate(template="my template {text}"),
    )
    extractor = config._get_extractor()
    assert isinstance(extractor, LLMEntityRelationExtractor)
    assert extractor.llm == llm
    assert extractor.on_error == OnError.IGNORE
    assert extractor.prompt_template.template == "my template {text}"


@patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 23, 0), False, False),
)
@patch(
    "neo4j_graphrag.experimental.pipeline.config.template_pipeline.simple_kg_builder.SimpleKGPipelineConfig.get_default_neo4j_driver"
)
def test_simple_kg_pipeline_config_writer(
    mock_driver: Mock,
    _: Mock,
    driver: neo4j.Driver,
) -> None:
    mock_driver.return_value = driver
    config = SimpleKGPipelineConfig(
        neo4j_database="my_db",
    )
    writer = config._get_writer()
    assert isinstance(writer, Neo4jWriter)
    assert writer.driver == driver
    assert writer.neo4j_database == "my_db"


@patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 23, 0), False, False),
)
@patch("neo4j_graphrag.experimental.pipeline.config.object_config.ComponentType.parse")
def test_simple_kg_pipeline_config_writer_overwrite(
    mock_component_parse: Mock,
    _: Mock,
    driver: neo4j.Driver,
) -> None:
    my_writer_config = ComponentConfig(
        class_="",
    )
    my_writer = Neo4jWriter(driver, neo4j_database="my_db")
    mock_component_parse.return_value = my_writer
    config = SimpleKGPipelineConfig(
        kg_writer=my_writer_config,  # type: ignore
        neo4j_database="my_other_db",
    )
    writer: Neo4jWriter = config._get_writer()  # type: ignore
    assert writer == my_writer
    # database not changed:
    assert writer.neo4j_database == "my_db"


def test_simple_kg_pipeline_config_connections_from_pdf() -> None:
    config = SimpleKGPipelineConfig(
        from_pdf=True,
        perform_entity_resolution=False,
    )
    connections = config._get_connections()
    assert len(connections) == 5
    expected_connections = [
        ("pdf_loader", "splitter"),
        ("schema", "extractor"),
        ("splitter", "chunk_embedder"),
        ("chunk_embedder", "extractor"),
        ("extractor", "writer"),
    ]
    for actual, expected in zip(connections, expected_connections):
        assert (actual.start, actual.end) == expected


def test_simple_kg_pipeline_config_connections_from_text() -> None:
    config = SimpleKGPipelineConfig(
        from_pdf=False,
        perform_entity_resolution=False,
    )
    connections = config._get_connections()
    assert len(connections) == 4
    expected_connections = [
        ("schema", "extractor"),
        ("splitter", "chunk_embedder"),
        ("chunk_embedder", "extractor"),
        ("extractor", "writer"),
    ]
    for actual, expected in zip(connections, expected_connections):
        assert (actual.start, actual.end) == expected


def test_simple_kg_pipeline_config_connections_with_er() -> None:
    config = SimpleKGPipelineConfig(
        from_pdf=True,
        perform_entity_resolution=True,
    )
    connections = config._get_connections()
    assert len(connections) == 6
    expected_connections = [
        ("pdf_loader", "splitter"),
        ("schema", "extractor"),
        ("splitter", "chunk_embedder"),
        ("chunk_embedder", "extractor"),
        ("extractor", "writer"),
        ("writer", "resolver"),
    ]
    for actual, expected in zip(connections, expected_connections):
        assert (actual.start, actual.end) == expected


def test_simple_kg_pipeline_config_run_params_from_pdf_file_path() -> None:
    config = SimpleKGPipelineConfig(from_pdf=True)
    assert config.get_run_params({"file_path": "my_file"}) == {
        "pdf_loader": {"filepath": "my_file"}
    }


def test_simple_kg_pipeline_config_run_params_from_text_text() -> None:
    config = SimpleKGPipelineConfig(from_pdf=False)
    assert config.get_run_params({"text": "my text"}) == {
        "splitter": {"text": "my text"}
    }


def test_simple_kg_pipeline_config_run_params_from_pdf_text() -> None:
    config = SimpleKGPipelineConfig(from_pdf=True)
    with pytest.raises(PipelineDefinitionError) as excinfo:
        config.get_run_params({"text": "my text"})
    assert "Expected 'file_path' argument when 'from_pdf' is True" in str(excinfo)


def test_simple_kg_pipeline_config_run_params_from_text_file_path() -> None:
    config = SimpleKGPipelineConfig(from_pdf=False)
    with pytest.raises(PipelineDefinitionError) as excinfo:
        config.get_run_params({"file_path": "my file"})
    assert "Expected 'text' argument when 'from_pdf' is False" in str(excinfo)


def test_simple_kg_pipeline_config_run_params_no_file_no_text() -> None:
    config = SimpleKGPipelineConfig(from_pdf=False)
    with pytest.raises(PipelineDefinitionError) as excinfo:
        config.get_run_params({})
    assert (
        "Use either 'text' (when from_pdf=False) or 'file_path' (when from_pdf=True) argument."
        in str(excinfo)
    )


def test_simple_kg_pipeline_config_run_params_both_file_and_text() -> None:
    config = SimpleKGPipelineConfig(from_pdf=False)
    with pytest.raises(PipelineDefinitionError) as excinfo:
        config.get_run_params({"text": "my text", "file_path": "my file"})
    assert (
        "Use either 'text' (when from_pdf=False) or 'file_path' (when from_pdf=True) argument."
        in str(excinfo)
    )
