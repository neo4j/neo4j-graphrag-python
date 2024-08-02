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
from typing import Any

from neo4j_genai.kg_construction.schema import Entity, Relation, SchemaBuilder
from neo4j_genai.pipeline import Component, DataModel
from pydantic import BaseModel, validate_call

logging.basicConfig(level=logging.DEBUG)


class DocumentChunkModel(DataModel):
    chunks: list[str]


class DocumentChunker(Component):
    async def run(self, text: str) -> DocumentChunkModel:
        chunks = [t.strip() for t in text.split(".") if t.strip()]
        return DocumentChunkModel(chunks=chunks)


class EntityModel(BaseModel):
    label: str
    properties: dict[str, str]


class Neo4jGraph(DataModel):
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]


class ERExtractor(Component):
    async def _process_chunk(self, chunk: str, schema: str) -> dict[str, Any]:
        return {
            "entities": [{"label": "Person", "properties": {"name": "John Doe"}}],
            "relations": [],
        }

    async def run(self, chunks: list[str], schema: str) -> Neo4jGraph:
        tasks = [self._process_chunk(chunk, schema) for chunk in chunks]
        result = await asyncio.gather(*tasks)
        merged_result: dict[str, Any] = {"entities": [], "relations": []}
        for res in result:
            merged_result["entities"] += res["entities"]
            merged_result["relations"] += res["relations"]
        return Neo4jGraph(
            entities=merged_result["entities"], relations=merged_result["relations"]
        )


class WriterModel(DataModel):
    status: str
    entities: list[EntityModel]
    relations: list[EntityModel]


class Writer(Component):
    @validate_call
    async def run(self, graph: Neo4jGraph) -> WriterModel:
        entities = graph.entities
        relations = graph.relations
        return WriterModel(
            status="OK",
            entities=[EntityModel(**e) for e in entities],
            relations=[EntityModel(**r) for r in relations],
        )


if __name__ == "__main__":
    from neo4j_genai.pipeline import Pipeline

    # Instantiate Entity and Relation objects
    entities = [
        Entity(name="PERSON", type="str", description="An individual human being."),
        Entity(
            name="ORGANIZATION",
            type="str",
            description="A structured group of people with a common purpose.",
        ),
        Entity(
            name="AGE",
            type="int",
        ),
    ]
    relations = [
        Relation(name="EMPLOYED_BY", description="Indicates employment relationship."),
        Relation(
            name="ORGANIZED_BY",
            description="Indicates organization responsible for an event.",
        ),
        Relation(name="ATTENDED_BY", description="Indicates attendance at an event."),
    ]
    potential_schema = {
        "PERSON": ["EMPLOYED_BY", "ATTENDED_BY"],
        "ORGANIZATION": ["EMPLOYED_BY", "ORGANIZED_BY"],
    }

    # Set up the pipeline
    pipe = Pipeline()
    pipe.add_component("chunker", DocumentChunker())
    pipe.add_component("schema", SchemaBuilder())
    pipe.add_component("extractor", ERExtractor())
    pipe.add_component("writer", Writer())
    pipe.connect("chunker", "extractor", input_config={"chunks": "chunker.chunks"})
    pipe.connect("schema", "extractor", input_config={"schema": "schema"})
    pipe.connect(
        "extractor",
        "writer",
        input_config={"graph": "extractor"},
    )

    pipe_inputs = {
        "chunker": {
            "text": """Graphs are everywhere.
            GraphRAG is the future of Artificial Intelligence.
            Robots are already running the world."""
        },
        "schema": {
            "entities": entities,
            "relations": relations,
            "potential_schema": potential_schema,
        },
    }
    print(asyncio.run(pipe.run(pipe_inputs)))
