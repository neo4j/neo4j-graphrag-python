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
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
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
def harry_potter_text() -> str:
    with open(os.path.join(BASE_DIR, "data/harry_potter.txt"), "r") as f:
        text = f.read()
    return text


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

    # Create an instance of the SimpleKGPipeline
    kg_builder_text = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        entities=entities,
        relations=relations,
        potential_schema=potential_schema,
        from_pdf=False,
        on_error="RAISE",
    )

    # Run the knowledge graph building process with text input
    text_input = "John Doe lives in New York City."
    await kg_builder_text.run_async(text=text_input)
