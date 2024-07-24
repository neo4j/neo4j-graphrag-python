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
from pydantic import BaseModel, Field, create_model
from typing import Any, List
from neo4j_genai.pipeline import Component


class Entity(BaseModel):
    name: str
    type: str = Field(
        ..., description="Type of the entity's name field, represented as a string."
    )
    description: str


class Relation(BaseModel):
    name: str
    description: str


class SchemaBuilder(Component):
    @staticmethod
    def create_schema_model(
        entities: List[Entity],
        relations: List[Relation],
        potential_schema: dict[str, List[str]],
    ) -> BaseModel:
        EntityModel = create_model(
            "EntityModel",
            **{
                entity.name: (
                    dict[str, Any],
                    Field(..., description=entity.description),
                )
                for entity in entities
            },
        )
        RelationModel = create_model(
            "RelationModel",
            **{
                relation.name: (
                    dict[str, str],
                    Field(..., description=relation.description),
                )
                for relation in relations
            },
        )

        SchemaModel = create_model(
            "SchemaModel",
            entities=(
                EntityModel,
                Field(
                    ...,
                    description="Templates for entities involved in the knowledge graph",
                ),
            ),
            relations=(
                RelationModel,
                Field(
                    ...,
                    description="Templates for relations defined in the knowledge graph",
                ),
            ),
            potential_schema=(
                dict[str, List[str]],
                Field(
                    ..., description="Schema outlining possible entity relationships"
                ),
            ),
        )
        schema_instance = SchemaModel(
            entities={
                entity.name: {
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                }
                for entity in entities
            },
            relations={
                relation.name: {
                    "name": relation.name,
                    "description": relation.description,
                }
                for relation in relations
            },
            potential_schema=potential_schema,
        )
        return schema_instance

    async def run(
        self,
        entities: List[Entity],
        relations: List[Relation],
        potential_schema: dict[str, List[str]],
    ) -> dict[str, Any]:
        schema_model = self.create_schema_model(entities, relations, potential_schema)
        return {"schema": schema_model}
