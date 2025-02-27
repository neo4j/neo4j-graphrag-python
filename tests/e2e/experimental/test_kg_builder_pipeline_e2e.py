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
from collections import Counter
from unittest.mock import MagicMock

import neo4j
import pytest
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.resolver import (
    SinglePropertyExactMatchResolver,
)
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
)
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm import LLMInterface, LLMResponse

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
def text_splitter() -> FixedSizeSplitter:
    return FixedSizeSplitter(chunk_size=500, chunk_overlap=10)


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
def entity_resolver(driver: neo4j.Driver) -> SinglePropertyExactMatchResolver:
    return SinglePropertyExactMatchResolver(driver)


@pytest.fixture
def kg_builder_pipeline(
    text_splitter: FixedSizeSplitter,
    chunk_embedder: TextChunkEmbedder,
    schema_builder: SchemaBuilder,
    entity_relation_extractor: LLMEntityRelationExtractor,
    kg_writer: Neo4jWriter,
    entity_resolver: SinglePropertyExactMatchResolver,
) -> Pipeline:
    pipe = Pipeline()
    # define the components
    pipe.add_component(text_splitter, "splitter")
    pipe.add_component(chunk_embedder, "embedder")
    pipe.add_component(schema_builder, "schema")
    pipe.add_component(entity_relation_extractor, "extractor")
    pipe.add_component(kg_writer, "writer")
    pipe.add_component(entity_resolver, "resolver")
    # define the execution order of component
    # and how the output of previous components must be used
    pipe.connect("splitter", "embedder", input_config={"text_chunks": "splitter"})
    # pipe.connect("splitter", "extractor", input_config={"chunks": "splitter"})
    pipe.connect("schema", "extractor", input_config={"schema": "schema"})
    pipe.connect("embedder", "extractor", input_config={"chunks": "embedder"})
    pipe.connect(
        "extractor",
        "writer",
        input_config={"graph": "extractor"},
    )
    pipe.connect("writer", "resolver", {})
    return pipe


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_pipeline_builder_happy_path(
    harry_potter_text: str,
    llm: MagicMock,
    embedder: MagicMock,
    driver: neo4j.Driver,
    kg_builder_pipeline: Pipeline,
) -> None:
    """When everything works as expected, extracted entities, relations and text
    chunks must be in the DB
    """
    driver.execute_query("MATCH (n) DETACH DELETE n")
    embedder.embed_query.return_value = [1, 2, 3]
    llm.ainvoke.side_effect = [
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
        "splitter": {"text": harry_potter_text},
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
                    label="LED_BY",
                ),
                SchemaRelation(
                    label="DRINKS",
                ),
            ],
            "potential_schema": [
                ("Person", "KNOWS", "Person"),
                ("Person", "DRINKS", "Potion"),
                ("Person", "PART_OF", "Organization"),
                ("Organization", "LED_BY", "Person"),
            ],
        },
        "extractor": {"document_info": {"path": "my document path"}},
    }
    res = await kg_builder_pipeline.run(pipe_inputs)
    # llm must have been called for each chunk
    assert llm.ainvoke.call_count == 2
    # result must be success
    assert isinstance(res, PipelineResult)
    assert res.run_id is not None
    assert "resolver" in res.result
    # check component's results
    chunks = await kg_builder_pipeline.store.get_result_for_component(
        res.run_id, "splitter"
    )
    assert len(chunks["chunks"]) == 2
    graph = await kg_builder_pipeline.store.get_result_for_component(
        res.run_id, "extractor"
    )
    # 3 entities + 2 chunks + 1 document
    nodes = graph["nodes"]
    assert len(nodes) == 6
    label_counts = dict(Counter([n["label"] for n in nodes]))
    assert label_counts == {
        "Chunk": 2,
        "Document": 1,
        "Person": 2,
        "Organization": 1,
    }
    # 2 relationships between entities
    # + 3 rels between entities and their chunk
    # + 2 "NEXT_CHUNK" rels
    relationships = graph["relationships"]
    assert len(relationships) == 8
    type_counts = dict(Counter([r["type"] for r in relationships]))
    assert type_counts == {
        "FROM_CHUNK": 3,
        "FROM_DOCUMENT": 2,
        "KNOWS": 1,
        "LED_BY": 1,
        "NEXT_CHUNK": 1,
    }
    # then check content of neo4j db
    created_nodes = driver.execute_query("MATCH (n) RETURN n")
    assert len(created_nodes.records) == 6
    created_rels = driver.execute_query("MATCH ()-[r]->() RETURN r")
    assert len(created_rels.records) == 8

    created_chunks = driver.execute_query("MATCH (n:Chunk) RETURN n").records
    assert len(created_chunks) == 2
    for c in created_chunks:
        node = c.get("n")
        assert node.get("embedding") == [1, 2, 3]
        assert node.get("text") is not None


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_pipeline_builder_failing_chunk_raise(
    harry_potter_text: str,
    embedder: MagicMock,
    llm: MagicMock,
    driver: neo4j.Driver,
    kg_builder_pipeline: Pipeline,
) -> None:
    """If on_error is set to "RAISE", any issue with the entity/relation
    extractor should stop the process with an exception. Nothing should be
    added to the DB
    """
    driver.execute_query("MATCH (n) DETACH DELETE n")
    embedder.embed_query.return_value = [1, 2, 3]
    llm.ainvoke.side_effect = [
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
    ]

    # user input:
    # the initial text
    # and the list of entities and relations we are looking for
    pipe_inputs = {
        "splitter": {"text": harry_potter_text},
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
    harry_potter_text: str,
    llm: MagicMock,
    embedder: MagicMock,
    driver: neo4j.Driver,
    kg_builder_pipeline: Pipeline,
) -> None:
    """If on_error is set to "IGNORE", process must continue
    and nodes/relationships created for the chunks that succeeded
    """
    driver.execute_query("MATCH (n) DETACH DELETE n")
    embedder.embed_query.return_value = [1, 2, 3]
    llm.ainvoke.side_effect = [
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
    ]

    # user input:
    # the initial text
    # and the list of entities and relations we are looking for
    pipe_inputs = {
        "splitter": {"text": harry_potter_text},
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
    ).component.on_error = OnError.IGNORE  # type: ignore[attr-defined, unused-ignore]
    res = await kg_builder_pipeline.run(pipe_inputs)
    # llm must have been called for each chunk
    assert llm.ainvoke.call_count == 2
    # result must be success
    assert isinstance(res, PipelineResult)
    assert res.run_id is not None
    assert res.result == {
        "resolver": {"number_of_created_nodes": 3, "number_of_nodes_to_resolve": 3}
    }
    # check component's results
    chunks = await kg_builder_pipeline.store.get_result_for_component(
        res.run_id, "splitter"
    )
    assert len(chunks["chunks"]) == 2
    graph = await kg_builder_pipeline.store.get_result_for_component(
        res.run_id, "extractor"
    )
    # 3 entities + 2 chunks
    nodes = graph["nodes"]
    assert len(nodes) == 5
    label_counts = dict(Counter([n["label"] for n in nodes]))
    assert label_counts == {
        "Chunk": 2,
        "Person": 2,
        "Organization": 1,
    }
    # 2 relationships between entities
    # + 3 rels between entities and their chunk
    # + 1 "NEXT_CHUNK" rels
    relationships = graph["relationships"]
    assert len(relationships) == 6
    type_counts = dict(Counter([r["type"] for r in relationships]))
    assert type_counts == {"FROM_CHUNK": 3, "KNOWS": 1, "LED_BY": 1, "NEXT_CHUNK": 1}
    # then check content of neo4j db
    created_nodes = driver.execute_query("MATCH (n) RETURN n")
    assert len(created_nodes.records) == 5
    created_rels = driver.execute_query("MATCH ()-[r]->() RETURN r")
    assert len(created_rels.records) == 6


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_pipeline_builder_two_documents(
    harry_potter_text_part1: str,
    harry_potter_text_part2: str,
    embedder: MagicMock,
    llm: MagicMock,
    driver: neo4j.Driver,
    kg_builder_pipeline: Pipeline,
) -> None:
    """Run same pipeline on two documents. Check entity resolution.

    First document:
    2 chunks, entities Harry and The Order of the Phoenix, 1 relationship
    Second document:
    1 chunk, entities Harry and Alastor Mad-Eye Moody, 1 relationship

    Should create:
    1 document node
    3 chunk nodes
    3 entities (1 Harry + the other two)
    ==> 7 nodes
    3 relationships for lexical graph + 3 relationships for the entity graph
    ==> 6 relationships
    """
    driver.execute_query("MATCH (n) DETACH DELETE n")
    embedder.embed_query.return_value = [1, 2, 3]
    llm.ainvoke.side_effect = [
        LLMResponse(
            content="""{
                        "nodes": [
                            {
                                "id": "0",
                                "label": "Person",
                                "properties": {
                                    "name": "Harry"
                                }
                            },
                            {
                                "id": "1",
                                "label": "Organization",
                                "properties": {
                                    "name": "The Order of the Phoenix"
                                }
                            }
                        ],
                        "relationships": [
                            {
                                "type": "MEMBER_OF",
                                "start_node_id": "0",
                                "end_node_id": "1"
                            }
                        ]
                    }"""
        ),
        LLMResponse(
            content="""{
                        "nodes": [
                            {
                                "id": "10",
                                "label": "Person",
                                "properties": {
                                    "name": "Harry"
                                }
                            },
                            {
                                "id": "11",
                                "label": "Person",
                                "properties": {
                                    "name": "Alastor Mad-Eye Moody"
                                }
                            }
                        ],
                        "relationships": [
                            {
                                "type": "KNOWS",
                                "start_node_id": "10",
                                "end_node_id": "11"
                            }
                        ]
                    }"""
        ),
        LLMResponse(content='{"nodes": [], "relationships": []}'),
    ]

    # user input:
    # the initial text
    # and the list of entities and relations we are looking for
    pipe_inputs_1 = {
        "splitter": {"text": harry_potter_text_part1},
        # note: schema not used in this test because
        # we are mocking the LLM
        "schema": {
            "entities": [],
            "relations": [],
            "potential_schema": [],
        },
    }
    pipe_inputs_2 = {
        "splitter": {"text": harry_potter_text_part2},
        # note: schema not used in this test because
        # we are mocking the LLM
        "schema": {
            "entities": [],
            "relations": [],
            "potential_schema": [],
        },
    }
    await kg_builder_pipeline.run(pipe_inputs_1)
    await kg_builder_pipeline.run(pipe_inputs_2)
    created_nodes = driver.execute_query("MATCH (n:__Entity__) RETURN n")
    assert len(created_nodes.records) == 3
    created_rels = driver.execute_query(
        "MATCH (:__Entity__)-[r]->(:__Entity__) RETURN r"
    )
    assert len(created_rels.records) == 2
