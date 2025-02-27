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
import neo4j
import pytest
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.experimental.components.types import (
    LexicalGraphConfig,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.experimental.pipeline import Pipeline


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
    assert result.result == {
        "writer": {
            "status": "SUCCESS",
            "metadata": {"node_count": 1, "relationship_count": 0},
        }
    }
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
