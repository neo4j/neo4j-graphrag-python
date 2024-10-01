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

import asyncio
import logging

import neo4j
from neo4j_graphrag.experimental.components.schema import (
    SchemaEntity,
    SchemaRelation,
)
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.experimental.pipeline.kg_builder import KnowledgeGraphBuilder
from neo4j_graphrag.experimental.components.entity_relation_extractor import OnError

logging.basicConfig(level=logging.INFO)


async def main(neo4j_driver: neo4j.Driver) -> PipelineResult:
    # Instantiate Entity and Relation objects
    entities = [
        SchemaEntity(label="PERSON", description="An individual human being."),
        SchemaEntity(
            label="ORGANIZATION",
            description="A structured group of people with a common purpose.",
        ),
        SchemaEntity(label="LOCATION", description="A location or place."),
        SchemaEntity(
            label="HORCRUX",
            description="A magical item in the Harry Potter universe.",
        ),
    ]
    relations = [
        SchemaRelation(
            label="SITUATED_AT", description="Indicates the location of a person."
        ),
        SchemaRelation(
            label="LED_BY",
            description="Indicates the leader of an organization.",
        ),
        SchemaRelation(
            label="OWNS",
            description="Indicates the ownership of an item such as a Horcrux.",
        ),
        SchemaRelation(
            label="INTERACTS", description="The interaction between two people."
        ),
    ]
    potential_schema = [
        ("PERSON", "SITUATED_AT", "LOCATION"),
        ("PERSON", "INTERACTS", "PERSON"),
        ("PERSON", "OWNS", "HORCRUX"),
        ("ORGANIZATION", "LED_BY", "PERSON"),
    ]

    # Instantiate the LLM
    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
        },
    )

    # Create an instance of the KnowledgeGraphBuilder
    kg_builder = KnowledgeGraphBuilder(
        llm=llm,
        driver=neo4j_driver,
        file_path="examples/pipeline/Harry Potter and the Death Hallows Summary.pdf",
        entities=entities,
        relations=relations,
        potential_schema=potential_schema,
        text_splitter=None,
        on_error=OnError.RAISE,
    )

    # Run the knowledge graph building process asynchronously
    result = await kg_builder.run_async()
    return result


if __name__ == "__main__":
    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    ) as driver:
        print(asyncio.run(main(driver)))
