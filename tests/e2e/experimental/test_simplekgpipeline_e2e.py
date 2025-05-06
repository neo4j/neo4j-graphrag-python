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

from typing import Any
from unittest.mock import MagicMock

import neo4j
import pytest
from neo4j import Driver
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import LLMResponse


@pytest.fixture(scope="function", autouse=True)
def clear_db(driver: Driver) -> Any:
    driver.execute_query("MATCH (n) DETACH DELETE n")
    yield


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_pipeline_builder_happy_path(
    harry_potter_text: str,
    llm: MagicMock,
    embedder: MagicMock,
    driver: neo4j.Driver,
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

    # Instantiate Entity and Relation objects
    entities = ["PERSON", "ORGANIZATION", "HORCRUX", "LOCATION"]
    relations = ["SITUATED_AT", "INTERACTS", "OWNS", "LED_BY"]
    potential_schema = [
        ("PERSON", "SITUATED_AT", "LOCATION"),
        ("PERSON", "INTERACTS", "PERSON"),
        ("PERSON", "OWNS", "HORCRUX"),
        ("ORGANIZATION", "LED_BY", "PERSON"),
    ]

    # Additional arguments
    lexical_graph_config = LexicalGraphConfig(chunk_node_label="chunkNodeLabel")
    from_pdf = False
    on_error = "RAISE"

    # Create an instance of the SimpleKGPipeline
    kg_builder_text = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        entities=entities,
        relations=relations,
        potential_schema=potential_schema,
        from_pdf=from_pdf,
        on_error=on_error,
        lexical_graph_config=lexical_graph_config,
    )

    # Run the knowledge graph building process with text input
    await kg_builder_text.run_async(text=harry_potter_text)


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_pipeline_builder_two_documents(
    harry_potter_text_part1: str,
    harry_potter_text_part2: str,
    llm: MagicMock,
    embedder: MagicMock,
    driver: neo4j.Driver,
) -> None:
    """When everything works as expected, extracted entities, relations and text
    chunks must be in the DB
    """
    driver.execute_query("MATCH (n) DETACH DELETE n")
    embedder.embed_query.return_value = [1, 2, 3]
    llm.ainvoke.side_effect = [
        # first document
        # first chunk
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
                        ],
                        "relationships": []
                    }"""
        ),
        # second chunk
        LLMResponse(content='{"nodes": [], "relationships": []}'),
        # second document
        # first chunk
        LLMResponse(
            content="""{
                        "nodes": [
                            {
                                "id": "0",
                                "label": "Person",
                                "properties": {
                                    "name": "Hermione Granger"
                                }
                            },
                        ],
                        "relationships": []
                    }"""
        ),
        # second chunk
        LLMResponse(content='{"nodes": [], "relationships": []}'),
    ]

    # Create an instance of the SimpleKGPipeline
    kg_builder_text = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        from_pdf=False,
        # provide minimal schema to bypass automatic schema extraction
        entities=["Person"],
        # in order to have 2 chunks:
        text_splitter=FixedSizeSplitter(chunk_size=400, chunk_overlap=5),
    )

    # Run the knowledge graph building process with text input
    await kg_builder_text.run_async(text=harry_potter_text_part1)
    await kg_builder_text.run_async(text=harry_potter_text_part2)

    # check graph content
    # check lexical graph content
    records, _, _ = driver.execute_query(
        "MATCH (start:Chunk)-[rel:NEXT_CHUNK]->(end:Chunk) RETURN start, rel, end"
    )
    assert len(records) == 2  # one for each run

    # check entity -> chunk relationships
    records, _, _ = driver.execute_query(
        "MATCH (chunk:Chunk)<-[rel:FROM_CHUNK]-(entity:__Entity__) RETURN chunk, rel, entity"
    )
    assert len(records) == 2  # two entities according to mocked LLMResponse


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_pipeline_builder_same_document_two_runs(
    harry_potter_text_part1: str,
    llm: MagicMock,
    embedder: MagicMock,
    driver: neo4j.Driver,
) -> None:
    """When everything works as expected, extracted entities, relations and text
    chunks must be in the DB
    """
    driver.execute_query("MATCH (n) DETACH DELETE n")
    embedder.embed_query.return_value = [1, 2, 3]
    llm.ainvoke.side_effect = [
        # first run
        # first chunk
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
                        ],
                        "relationships": []
                    }"""
        ),
        # second chunk
        LLMResponse(content='{"nodes": [], "relationships": []}'),
        # second run
        # first chunk
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
                        ],
                        "relationships": []
                    }"""
        ),
        # second chunk
        LLMResponse(content='{"nodes": [], "relationships": []}'),
    ]

    # Create an instance of the SimpleKGPipeline
    kg_builder_text = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        from_pdf=False,
        # provide minimal schema to bypass automatic schema extraction
        entities=["Person"],
        # in order to have 2 chunks:
        text_splitter=FixedSizeSplitter(chunk_size=400, chunk_overlap=5),
    )

    # Run the knowledge graph building process with text input
    await kg_builder_text.run_async(text=harry_potter_text_part1)
    await kg_builder_text.run_async(text=harry_potter_text_part1)

    # check lexical graph content
    records, _, _ = driver.execute_query(
        "MATCH (start:Chunk)-[rel:NEXT_CHUNK]->(end:Chunk) RETURN start, rel, end"
    )
    assert len(records) == 2  # one for each run

    # check entity -> chunk relationships
    records, _, _ = driver.execute_query(
        "MATCH (chunk:Chunk)<-[rel:FROM_CHUNK]-(entity:__Entity__) RETURN chunk, rel, entity"
    )
    assert len(records) == 2  # two entities according to mocked LLMResponse


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_pipeline_builder_with_automatic_schema_extraction(
    harry_potter_text_part1: str,
    llm: MagicMock,
    embedder: MagicMock,
    driver: neo4j.Driver,
) -> None:
    """Test pipeline with automatic schema extraction (no schema provided).
    This test verifies that the pipeline correctly handles automatic schema extraction.
    """
    driver.execute_query("MATCH (n) DETACH DELETE n")
    embedder.embed_query.return_value = [1, 2, 3]

    # set up mock LLM responses for both schema extraction and entity extraction
    llm.ainvoke.side_effect = [
        # first call - schema extraction response
        LLMResponse(
            content="""{
                "entities": [
                    {
                        "label": "Person",
                        "description": "A character in the story",
                        "properties": [
                            {"name": "name", "type": "STRING"},
                            {"name": "age", "type": "INTEGER"}
                        ]
                    },
                    {
                        "label": "Location",
                        "description": "A place in the story",
                        "properties": [
                            {"name": "name", "type": "STRING"}
                        ]
                    }
                ],
                "relations": [
                    {
                        "label": "LOCATED_AT",
                        "description": "Indicates where a person is located",
                        "properties": []
                    }
                ],
                "potential_schema": [
                    ["Person", "LOCATED_AT", "Location"]
                ]
            }"""
        ),
        # second call - entity extraction for first chunk
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
                        "label": "Location",
                        "properties": {
                            "name": "Hogwarts"
                        }
                    }
                ],
                "relationships": [
                    {
                        "type": "LOCATED_AT",
                        "start_node_id": "0",
                        "end_node_id": "1"
                    }
                ]
            }"""
        ),
        # third call - entity extraction for second chunk (if text is split)
        LLMResponse(content='{"nodes": [], "relationships": []}'),
    ]

    # create an instance of the SimpleKGPipeline with NO schema provided
    kg_builder_text = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        from_pdf=False,
        # use smaller chunk size to ensure we have at least 2 chunks
        text_splitter=FixedSizeSplitter(chunk_size=400, chunk_overlap=5),
    )

    # run the knowledge graph building process with text input
    await kg_builder_text.run_async(text=harry_potter_text_part1)

    # verify LLM was called for schema extraction
    assert llm.ainvoke.call_count >= 2

    # verify entities were created
    records, _, _ = driver.execute_query("MATCH (n:Person) RETURN n")
    assert len(records) == 1

    # verify locations were created
    records, _, _ = driver.execute_query("MATCH (n:Location) RETURN n")
    assert len(records) == 1

    # verify relationships were created
    records, _, _ = driver.execute_query(
        "MATCH (p:Person)-[r:LOCATED_AT]->(l:Location) RETURN p, r, l"
    )
    assert len(records) == 1

    # verify chunks and relationships to entities
    records, _, _ = driver.execute_query(
        "MATCH (c:Chunk)<-[:FROM_CHUNK]-(e) RETURN c, e"
    )
    assert len(records) >= 1
