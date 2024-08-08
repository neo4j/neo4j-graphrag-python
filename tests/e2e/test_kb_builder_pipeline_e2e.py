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

import os
from unittest.mock import MagicMock

import neo4j
import pytest
from langchain_text_splitters import CharacterTextSplitter
from neo4j_genai.components.embedder import TextChunkEmbedder
from neo4j_genai.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_genai.components.kg_writer import Neo4jWriter
from neo4j_genai.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
)
from neo4j_genai.components.text_splitters.langchain import LangChainTextSplitterAdapter
from neo4j_genai.embedder import Embedder
from neo4j_genai.exceptions import LLMGenerationError
from neo4j_genai.llm import LLMInterface, LLMResponse
from neo4j_genai.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def llm() -> LLMInterface:
    llm = MagicMock(spec=LLMInterface)
    return llm


@pytest.fixture
def embedder() -> Embedder:
    embedder = MagicMock(spec=Embedder)
    return embedder


@pytest.fixture
def schema_builder() -> SchemaBuilder:
    return SchemaBuilder()


@pytest.fixture
def text_splitter() -> LangChainTextSplitterAdapter:
    return LangChainTextSplitterAdapter(
        # chunk_size=50 for the sake of this demo
        CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator="\n\n")
    )


@pytest.fixture
def chunk_embedder(embedder: Embedder) -> TextChunkEmbedder:
    return TextChunkEmbedder(embedder=embedder)


@pytest.fixture
def entity_relation_extractor(llm: LLMInterface) -> LLMEntityRelationExtractor:
    return LLMEntityRelationExtractor(
        llm=llm,
        on_error=OnError.RAISE,
    )


@pytest.fixture
def kg_writer(driver: neo4j.Driver) -> Neo4jWriter:
    return Neo4jWriter(driver)


@pytest.fixture
def kg_builder_pipeline(
    text_splitter: LangChainTextSplitterAdapter,
    chunk_embedder: TextChunkEmbedder,
    schema_builder: SchemaBuilder,
    entity_relation_extractor: LLMEntityRelationExtractor,
    kg_writer: Neo4jWriter,
) -> Pipeline:
    pipe = Pipeline()
    # define the components
    pipe.add_component(
        "splitter",
        text_splitter,
    )
    pipe.add_component("embedder", chunk_embedder)
    pipe.add_component("schema", schema_builder)
    pipe.add_component(
        "extractor",
        entity_relation_extractor,
    )
    pipe.add_component("writer", kg_writer)
    # define the execution order of component
    # and how the output of previous components must be used
    pipe.connect("splitter", "embedder", input_config={"text_chunks": "splitter"})
    pipe.connect("schema", "extractor", input_config={"schema": "schema"})
    pipe.connect("embedder", "extractor", input_config={"chunks": "embedder"})
    pipe.connect(
        "extractor",
        "writer",
        input_config={"graph": "extractor"},
    )
    return pipe


@pytest.fixture
def text() -> str:
    with open(os.path.join(BASE_DIR, "data/harry_potter.txt", "r")) as f:
        text = f.read()
    return text


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_pipeline_builder_happy_path(
    text: str,
    llm: MagicMock,
    embedder: MagicMock,
    driver: neo4j.Driver,
    kg_builder_pipeline: Pipeline,
) -> None:
    """When everything works as expected, extracted entities, relations and text
    chunks must be in the DB
    """
    embedder.embed.return_value = [1, 2, 3]
    llm.invoke.side_effect = [
        LLMResponse(
            content="""{
                        "nodes": [
                            {
                                "id": "0",
                                "label": "Person",
                                "properties": {
                                    "name": "Harry Potter"
                                }
                            },
                            {
                                "id": "1",
                                "label": "Person",
                                "properties": {
                                    "name": "Alastor Mad-Eye Moody"
                                }
                            },
                            {
                                "id": "2",
                                "label": "Organization",
                                "properties": {
                                    "name": "The Order of the Phoenix"
                                }
                            }
                        ],
                        "relationships": [
                            {
                                "type": "KNOWS",
                                "start_node_id": "0",
                                "end_node_id": "1"
                            },
                            {
                                "type": "LED_BY",
                                "start_node_id": "2",
                                "end_node_id": "1"
                            }
                        ]
                    }"""
        ),
        LLMResponse(content='{"nodes": [], "relationships": []}'),
        LLMResponse(content='{"nodes": [], "relationships": []}'),
    ]

    # user input:
    # the initial text
    # and the list of entities and relations we are looking for
    pipe_inputs = {
        "splitter": {"text": text},
        "schema": {
            "entities": [
                SchemaEntity(
                    label="Person",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                        SchemaProperty(name="place_of_birth", type="STRING"),
                        SchemaProperty(name="date_of_birth", type="DATE"),
                    ],
                ),
                SchemaEntity(
                    label="Organization",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                    ],
                ),
                SchemaEntity(
                    label="Potion",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                    ],
                ),
                SchemaEntity(
                    label="Location",
                    properties=[
                        SchemaProperty(name="address", type="STRING"),
                    ],
                ),
            ],
            "relations": [
                SchemaRelation(
                    label="KNOWS",
                ),
                SchemaRelation(
                    label="PART_OF",
                ),
                SchemaRelation(
                    label="LEAD_BY",
                ),
                SchemaRelation(
                    label="DRINKS",
                ),
            ],
            "potential_schema": [
                ("Person", "KNOWS", "Person"),
                ("Person", "DRINKS", "Potion"),
                ("Person", "PART_OF", "Organization"),
                ("Organization", "LEAD_BY", "Person"),
            ],
        },
    }
    res = await kg_builder_pipeline.run(pipe_inputs)
    # llm must have been called for each chunk
    assert llm.invoke.call_count == 3
    # result must be success
    assert res == {"writer": {"status": "SUCCESS"}}
    # check component's results
    chunks = kg_builder_pipeline.get_results_for_component("splitter")
    assert len(chunks["chunks"]) == 3
    graph = kg_builder_pipeline.get_results_for_component("extractor")
    # 3 entities + 3 chunks
    assert len(graph["nodes"]) == 6
    # 2 relationships between entities
    # + 3 rels between entities and their chunk
    # + 2 "NEXT_CHUNK" rels
    assert len(graph["relationships"]) == 7
    # then check content of neo4j db
    created_nodes = driver.execute_query("MATCH (n) RETURN n")
    assert len(created_nodes.records) == 6
    created_rels = driver.execute_query("MATCH ()-[r]->() RETURN r")
    assert len(created_rels.records) == 7

    created_chunks = driver.execute_query("MATCH (n:Chunk) RETURN n").records
    assert len(created_chunks) == 3
    for c in created_chunks:
        assert c.get("embedding") == [1, 2, 3]
        assert c.get("text") is not None


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_pipeline_builder_failing_chunk_raise(
    text: str,
    llm: MagicMock,
    driver: neo4j.Driver,
    kg_builder_pipeline: Pipeline,
) -> None:
    """If on_error is set to "RAISE", any issue with the entity/relation
    extractor should stop the process with an exception. Nothing should be
    added to the DB
    """
    llm.invoke.side_effect = [
        LLMResponse(
            content="""{
                        "nodes": [
                            {
                                "id": "0",
                                "label": "Person",
                                "properties": {
                                    "name": "Harry Potter"
                                }
                            },
                            {
                                "id": "1",
                                "label": "Person",
                                "properties": {
                                    "name": "Alastor Mad-Eye Moody"
                                }
                            },
                            {
                                "id": "2",
                                "label": "Organization",
                                "properties": {
                                    "name": "The Order of the Phoenix"
                                }
                            }
                        ],
                        "relationships": [
                            {
                                "type": "KNOWS",
                                "start_node_id": "0",
                                "end_node_id": "1"
                            },
                            {
                                "type": "LED_BY",
                                "start_node_id": "2",
                                "end_node_id": "1"
                            }
                        ]
                    }"""
        ),
        LLMResponse(content="invalid json"),
        LLMResponse(content='{"nodes": [], "relationships": []}'),
    ]

    # user input:
    # the initial text
    # and the list of entities and relations we are looking for
    pipe_inputs = {
        "splitter": {"text": text},
        # note: schema not used in this test because
        # we are mocking the LLM
        "schema": {
            "entities": [],
            "relations": [],
            "potential_schema": [],
        },
    }
    with pytest.raises(LLMGenerationError):
        await kg_builder_pipeline.run(pipe_inputs)
    created_nodes = driver.execute_query("MATCH (n) RETURN n")
    assert len(created_nodes.records) == 0
    created_rels = driver.execute_query("MATCH ()-[r]->() RETURN r")
    assert len(created_rels.records) == 0


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_pipeline_builder_failing_chunk_do_not_raise(
    text: str,
    llm: MagicMock,
    driver: neo4j.Driver,
    kg_builder_pipeline: Pipeline,
) -> None:
    """If on_error is set to "IGNORE", process must continue
    and nodes/relationships created for the chunks that succeeded
    """
    llm.invoke.side_effect = [
        LLMResponse(content="invalid json"),
        LLMResponse(
            content="""{
                        "nodes": [
                            {
                                "id": "0",
                                "label": "Person",
                                "properties": {
                                    "name": "Harry Potter"
                                }
                            },
                            {
                                "id": "1",
                                "label": "Person",
                                "properties": {
                                    "name": "Alastor Mad-Eye Moody"
                                }
                            },
                            {
                                "id": "2",
                                "label": "Organization",
                                "properties": {
                                    "name": "The Order of the Phoenix"
                                }
                            }
                        ],
                        "relationships": [
                            {
                                "type": "KNOWS",
                                "start_node_id": "0",
                                "end_node_id": "1"
                            },
                            {
                                "type": "LED_BY",
                                "start_node_id": "2",
                                "end_node_id": "1"
                            }
                        ]
                    }"""
        ),
        LLMResponse(content='{"nodes": [], "relationships": []}'),
    ]

    # user input:
    # the initial text
    # and the list of entities and relations we are looking for
    pipe_inputs = {
        "splitter": {"text": text},
        # note: schema not used in this test because
        # we are mocking the LLM
        "schema": {
            "entities": [],
            "relations": [],
            "potential_schema": [],
        },
    }
    kg_builder_pipeline.get_node_by_name(
        "extractor"
    ).component.on_error = OnError.IGNORE  # type: ignore[arg-type, unused-ignore]
    res = await kg_builder_pipeline.run(pipe_inputs)
    # llm must have been called for each chunk
    assert llm.return_value.invoke.call_count == 3
    # result must be success
    assert res == {"writer": {"status": "SUCCESS"}}
    # check component's results
    chunks = kg_builder_pipeline.get_results_for_component("splitter")
    assert len(chunks["chunks"]) == 3
    graph = kg_builder_pipeline.get_results_for_component("extractor")
    # 3 entities + 3 chunks
    assert len(graph["nodes"]) == 6
    # 2 relationships between entities
    # + 3 rels between entities and their chunk
    # + 2 "NEXT_CHUNK" rels
    assert len(graph["relationships"]) == 7
    # then check content of neo4j db
    created_nodes = driver.execute_query("MATCH (n) RETURN n")
    assert len(created_nodes.records) == 6
    created_rels = driver.execute_query("MATCH ()-[r]->() RETURN r")
    assert len(created_rels.records) == 7