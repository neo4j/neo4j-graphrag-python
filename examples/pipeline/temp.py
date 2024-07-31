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
from typing import Any

from neo4j_genai.pipeline import Component


class DocumentChunker(Component):
    async def run(self, text: str) -> dict[str, Any]:
        return {"chunks": [t.strip() for t in text.split(".") if t.strip()]}


class ERExtractor(Component):
    async def _process_chunk(self, chunk: str, schema: str) -> dict[str, Any]:
        return {
            "data": {
                "entities": [{"label": "Person", "properties": {"name": "John Doe"}}],
                "relations": [],
            }
        }

    async def run(self, chunks: list[str], schema: str) -> dict[str, Any]:
        tasks = [self._process_chunk(chunk, schema) for chunk in chunks]
        result = await asyncio.gather(*tasks)
        merged_result: dict[str, Any] = {"data": {"entities": [], "relations": []}}
        for res in result:
            merged_result["data"]["entities"] += res["data"]["entities"]
            merged_result["data"]["relations"] += res["data"]["relations"]
        return merged_result


class Writer(Component):
    async def run(
        self, entities: dict[str, Any], relations: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "status": "OK",
            "entities": entities,
            "relations": relations,
        }


if __name__ == "__main__":
    from neo4j_genai.kg_construction.schema import Entity, Relation, SchemaBuilder
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
            name="AGE", type="int", description="The age of an individual in years."
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
        "EVENT": ["ORGANIZED_BY", "ATTENDED_BY"],
    }

    # Set up the pipeline
    schema_builder = SchemaBuilder()
    extractor = ERExtractor()
    writer = Writer()
    chunker = DocumentChunker()

    pipe = Pipeline()
    pipe.add_component("chunker", chunker)
    pipe.add_component("schema", schema_builder)
    pipe.add_component("extractor", extractor)
    pipe.add_component("writer", writer)
    pipe.connect("chunker", "extractor", input_defs={"chunks": "chunker.chunks"})
    pipe.connect("schema", "extractor", input_defs={"schema": "schema.schema"})
    pipe.connect(
        "extractor",
        "writer",
        input_defs={
            "entities": "extractor.data.entities",
            "relations": "extractor.data.relations",
        },
    )

    pipe_inputs = {
        "chunker": {
            "text": "Graphs are everywhere. GraphRAG is the future of Artificial Intelligence. Robots are already running the world."
        },
        "schema": {
            "entities": entities,
            "relations": relations,
            "potential_schema": potential_schema,
        },
    }

    # Run the pipeline
    print(asyncio.run(pipe.run(pipe_inputs)))
