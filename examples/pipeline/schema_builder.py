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
from typing import Any, List, Dict

from neo4j_genai.core.pipeline import Component
from pydantic import BaseModel, Field, create_model


class Entity(BaseModel):
    name: str
    type: str = Field(...,
                      description="Type of the entity's name field, represented as a string.")
    description: str


class Relation(BaseModel):
    name: str
    description: str


class DocumentChunker(Component):
    async def run(self, text: str) -> dict[str, Any]:
        return {"chunks": [t.strip() for t in text.split(".") if t.strip()]}


class SchemaBuilder(Component):
    def create_schema_model(self, entities: List[Entity], relations: List[Relation],
                            potential_schema: Dict[str, List[str]]) -> BaseModel:
        """
        Creates a dynamic Pydantic model based on provided entity and relation templates,
        including a schema of potential relations.
        """
        EntityModel = create_model('EntityModel', **{
            entity.name: (Dict[str, Any], Field(..., description=entity.description))
            for entity in entities
        })
        RelationModel = create_model('RelationModel', **{
            relation.name: (
            Dict[str, str], Field(..., description=relation.description))
            for relation in relations
        })

        SchemaModel = create_model('SchemaModel',
                                   entities=(EntityModel, Field(...,
                                                                description="Templates for entities involved in the knowledge graph")),
                                   relations=(RelationModel, Field(...,
                                                                   description="Templates for relations defined in the knowledge graph")),
                                   potential_schema=(Dict[str, List[str]], Field(...,
                                                                                 description="Schema outlining possible entity relationships"))
                                   )

        schema_instance = SchemaModel(
            entities=EntityModel(**{
                entity.name: {
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description
                } for entity in entities
            }),
            relations=RelationModel(**{
                relation.name: {
                    "name": relation.name,
                    "description": relation.description
                } for relation in relations
            }),
            potential_schema=potential_schema
        )
        return schema_instance

    async def run(self, schema_input: dict[str, Any]) -> dict[str, Any]:
        entities = [Entity(**e) for e in schema_input["entities"]]
        relations = [Relation(**r) for r in schema_input["relations"]]
        potential_schema = schema_input["potential_schema"]

        schema_model = self.create_schema_model(entities, relations, potential_schema)
        return {"schema": schema_model}


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
    from neo4j_genai.core.pipeline import Pipeline

    # Setup components
    schema_builder = SchemaBuilder()
    chunker = DocumentChunker()
    extractor = ERExtractor()
    writer = Writer()

    # Create a pipeline instance
    pipe = Pipeline()
    pipe.add_component("chunker", chunker)
    pipe.add_component("schema", schema_builder)
    pipe.add_component("extractor", extractor)
    pipe.add_component("writer", writer)

    # Ensure correct input_defs are used; "schema_input" should match how you intend to pass it
    pipe.connect("chunker", "extractor", input_defs={"chunks": "chunker.chunks"})
    pipe.connect("schema", "extractor",
                 input_defs={"schema": "schema.schema"})
    pipe.connect(
        "extractor",
        "writer",
        input_defs={
            "entities": "extractor.data.entities",
            "relations": "extractor.data.relations",
        },
    )

    # Example schema input setup
    schema_input = {
        "entities": [
            {"name": "PERSON", "type": "str",
             "description": "An individual human being."},
            {"name": "ORGANIZATION", "type": "str",
             "description": "A structured group of people with a common purpose."},
            {"name": "AGE", "type": "int",
             "description": "The age of an individual in years."},
        ],
        "relations": [
            {"name": "EMPLOYED_BY",
             "description": "Indicates employment relationship."},
            {"name": "ORGANIZED_BY",
             "description": "Indicates organization responsible for an event."},
            {"name": "ATTENDED_BY", "description": "Indicates attendance at an event."},
        ],
        "potential_schema": {
            "PERSON": ["EMPLOYED_BY", "ATTENDED_BY"],
            "ORGANIZATION": ["EMPLOYED_BY", "ORGANIZED_BY"],
            "EVENT": ["ORGANIZED_BY", "ATTENDED_BY"]
        }
    }

    # When running the pipeline, make sure the input to 'schema' is named correctly if being used directly
    pipe_inputs = {
        "chunker": {
            "text": "Graphs are everywhere. GraphRAG is the future of Artificial Intelligence. Robots are already running the world."
        },
        "schema": {"schema_input": schema_input}
    }

    # Execute the pipeline
    print(asyncio.run(pipe.run(pipe_inputs)))
