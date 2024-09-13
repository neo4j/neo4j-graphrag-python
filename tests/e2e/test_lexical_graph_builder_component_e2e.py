"""
What is tested?

1. Chunking -> LexicalGraph -> Writer
2. Chunking -> LexicalGraph -> Extractor(create_lexical_graph=True) -> Writer
3. Chunking -> Extractor(create_lexical_graph=True) -> Writer  # deprecated (tested in the pipeline e2e tests)
4. Reader -> Extractor -> Writer (tested in reader e2e tests)
"""

from unittest.mock import MagicMock

import neo4j
import pytest
from neo4j_genai.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_genai.experimental.components.kg_writer import Neo4jWriter
from neo4j_genai.experimental.components.lexical_graph import (
    LexicalGraphBuilder,
    LexicalGraphConfig,
)
from neo4j_genai.experimental.components.types import TextChunk, TextChunks
from neo4j_genai.experimental.pipeline import Pipeline
from neo4j_genai.llm import LLMResponse


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
async def test_lexical_graph_before_extractor(
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
    pipe.add_component(LexicalGraphBuilder(), "lexical_graph")
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
    created_persons = driver.execute_query(f"MATCH (n:Person) RETURN n")
    assert len(created_persons.records) == 2

    created_entity_to_chunk_rels = driver.execute_query(
        f"MATCH ()-[r:{default_config.node_to_chunk_relationship_type}]->() RETURN r"
    )
    assert len(created_entity_to_chunk_rels.records) == 2
