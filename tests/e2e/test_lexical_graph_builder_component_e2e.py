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
from unittest.mock import MagicMock

import neo4j
import pytest
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.lexical_graph import (
    LexicalGraphBuilder,
    LexicalGraphConfig,
)
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.llm import LLMResponse


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_lexical_graph_component_alone_default_config(
    driver: neo4j.Driver,
) -> None:
    pipe = Pipeline()
    pipe.add_component(LexicalGraphBuilder(), "lexical_graph")
    pipe.add_component(Neo4jWriter(driver), "writer")
    pipe.connect("lexical_graph", "writer", {"graph": "lexical_graph.graph"})

    result = await pipe.run(
        {
            "lexical_graph": {
                "text_chunks": TextChunks(chunks=[TextChunk(text="my text", index=0)])
            }
        }
    )
    assert result.result == {"writer": {"status": "SUCCESS"}}
    default_config = LexicalGraphConfig()
    created_chunks = driver.execute_query(
        f"MATCH (n:{default_config.chunk_node_label}) RETURN n"
    )
    assert len(created_chunks.records) == 1
    created_chunks = driver.execute_query(
        f"MATCH (n:{default_config.document_node_label}) RETURN n"
    )
    assert len(created_chunks.records) == 0
    created_rels = driver.execute_query("MATCH ()-[r]->() RETURN r")
    assert len(created_rels.records) == 0


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction")
async def test_lexical_graph_before_extractor_custom_prefix(
    driver: neo4j.Driver, llm: MagicMock
) -> None:
    driver.execute_query("MATCH (n) DETACH DELETE n")
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
                    ],
                    "relationships": [
                        {
                            "type": "KNOWS",
                            "start_node_id": "0",
                            "end_node_id": "1"
                        },
                    ]
                }"""
        ),
    ]
    pipe = Pipeline()
    pipe.add_component(
        LexicalGraphBuilder(config=LexicalGraphConfig(id_prefix="myPrefix")),
        "lexical_graph",
    )
    pipe.add_component(
        LLMEntityRelationExtractor(llm, create_lexical_graph=False), "extractor"
    )
    pipe.add_component(Neo4jWriter(driver), "lg_writer")  # lexical graph writer
    pipe.add_component(Neo4jWriter(driver), "eg_writer")  # entity graph writer
    chunks = TextChunks(chunks=[TextChunk(text="my text", index=0)])
    pipe.connect("lexical_graph", "lg_writer", {"graph": "lexical_graph.graph"})
    pipe.connect(
        "lexical_graph", "extractor", {"lexical_graph_config": "lexical_graph.config"}
    )
    pipe.connect("extractor", "eg_writer", {"graph": "extractor"})

    result = await pipe.run(
        {"lexical_graph": {"text_chunks": chunks}, "extractor": {"chunks": chunks}}
    )
    assert result.result == {
        "eg_writer": {"status": "SUCCESS"},
        "lg_writer": {"status": "SUCCESS"},
    }

    # check the lexical graph has been created
    default_config = LexicalGraphConfig()
    created_chunks = driver.execute_query(
        f"MATCH (n:{default_config.chunk_node_label}) RETURN n"
    )
    assert len(created_chunks.records) == 1
    created_documents = driver.execute_query(
        f"MATCH (n:{default_config.document_node_label}) RETURN n"
    )
    assert len(created_documents.records) == 0

    # check the entity graph has been created
    created_persons = driver.execute_query("MATCH (n:Person) RETURN n")
    assert len(created_persons.records) == 2

    created_entity_to_chunk_rels = driver.execute_query(
        f"MATCH ()-[r:{default_config.node_to_chunk_relationship_type}]->() RETURN r"
    )
    assert len(created_entity_to_chunk_rels.records) == 2
