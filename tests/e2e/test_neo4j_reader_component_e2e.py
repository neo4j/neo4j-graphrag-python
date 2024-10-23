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
from venv import create

import neo4j
import pytest
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.neo4j_reader import Neo4jChunkReader
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig, TextChunk
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.llm import LLMResponse


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction_with_chunks")
async def test_neo4j_reader(driver: neo4j.Driver) -> None:
    reader = Neo4jChunkReader(driver)
    res = await reader.run()
    assert len(res.chunks) == 2
    assert res.chunks[0] == TextChunk(
        index=0, text="some text", metadata={"id": "0", "embedding": None}
    )
    assert res.chunks[1] == TextChunk(
        index=1, text="some longer text", metadata={"id": "1", "embedding": None}
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction_with_chunks")
async def test_neo4j_reader_in_pipeline(driver: neo4j.Driver, llm: MagicMock) -> None:
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
                    }
                ],
                "relationships": []
            }"""
        ),
        LLMResponse(content='{"nodes": [], "relationships": []}'),
    ]
    pipeline = Pipeline()
    pipeline.add_component(Neo4jChunkReader(driver), "reader")
    pipeline.add_component(
        LLMEntityRelationExtractor(llm, create_lexical_graph=False), "extractor"
    )
    pipeline.connect("reader", "extractor", {"chunks": "reader"})
    pipeline_output = await pipeline.run({})
    created_graph = pipeline_output.result["extractor"]
    assert len(created_graph["nodes"]) == 1
    assert len(created_graph["relationships"]) == 0


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_neo4j_for_kg_construction_with_chunks")
async def test_neo4j_reader_in_pipeline_with_lexical_graph(
    driver: neo4j.Driver, llm: MagicMock
) -> None:
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
                    }
                ],
                "relationships": []
            }"""
        ),
        LLMResponse(content='{"nodes": [], "relationships": []}'),
    ]
    pipeline = Pipeline()
    pipeline.add_component(Neo4jChunkReader(driver), "reader")
    pipeline.add_component(
        LLMEntityRelationExtractor(llm, create_lexical_graph=False), "extractor"
    )
    pipeline.connect("reader", "extractor", {"chunks": "reader"})
    lg_config = LexicalGraphConfig()
    pipeline_output = await pipeline.run(
        {
            "extractor": {
                "lexical_graph_config": lg_config,
            }
        }
    )
    created_graph = pipeline_output.result["extractor"]
    assert len(created_graph["nodes"]) == 1
    assert len(created_graph["relationships"]) == 1  # entity to chunk relationship
    assert (
        created_graph["relationships"][0]["type"]
        == lg_config.node_to_chunk_relationship_type
    )
