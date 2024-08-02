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

from typing import Any, Dict, List

from neo4j_genai.exceptions import SchemaValidationError
from neo4j_genai.pipeline import Component, DataModel
from pydantic import BaseModel, Field, model_validator


class Entity(BaseModel):
    name: str
    type: str = Field(
        ..., description="Type of the entity's name field, represented as a string."
    )
    description: str


class Relation(BaseModel):
    name: str
    description: str


class SchemaConfig(DataModel):
    entities: Dict[str, Dict[str, Any]]
    relations: Dict[str, Dict[str, Any]]
    potential_schema: Dict[str, List[str]]

    @model_validator(mode="before")
    def check_schema(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        entities = data.get("entities", {}).keys()
        relations = data.get("relations", {}).keys()
        potential_schema = data.get("potential_schema", {})

        for entity in potential_schema.keys():
            if entity not in entities:
                raise SchemaValidationError(
                    f"Entity '{entity}' is not defined in the provided entities."
                )

        for rels in potential_schema.values():
            for rel in rels:
                if rel not in relations:
                    raise SchemaValidationError(
                        f"Relation '{rel}' is not defined in the provided relations."
                    )

        return data


class SchemaBuilder(Component):
    """
    A builder class for constructing SchemaConfig objects from given entities,
    relations, and their interrelationships defined in a potential schema.
    """

    @staticmethod
    def create_schema_model(
        entities: List[Entity],
        relations: List[Relation],
        potential_schema: Dict[str, List[str]],
    ) -> SchemaConfig:
        """
        Creates a SchemaConfig object from Lists of Entity and Relation objects
        and a Dictionary defining potential relationships.

        Args:
            entities (List[Entity]): List of Entity objects.
            relations (List[Relation]): List of Relation objects.
            potential_schema (Dict[str, List[str]]): Dictionary mapping entity names to Lists of relation names.

        Returns:
            SchemaConfig: A configured schema object.
        """
        entity_dict = {entity.name: entity.dict() for entity in entities}
        relation_dict = {relation.name: relation.dict() for relation in relations}

        return SchemaConfig(
            entities=entity_dict,
            relations=relation_dict,
            potential_schema=potential_schema,
        )

    async def run(
        self,
        entities: List[Entity],
        relations: List[Relation],
        potential_schema: Dict[str, List[str]],
    ) -> SchemaConfig:
        """
        Asynchronously constructs and returns a SchemaConfig object.

        Args:
            entities (List[Entity]): List of Entity objects.
            relations (List[Relation]): List of Relation objects.
            potential_schema (Dict[str, List[str]]): Dictionary mapping entity names to Lists of relation names.

        Returns:
            SchemaConfig: A configured schema object, constructed asynchronously.
        """
        return self.create_schema_model(entities, relations, potential_schema)
