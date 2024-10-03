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

from typing import Any, Dict, List, Literal, Tuple

from pydantic import BaseModel, ValidationError, model_validator, validate_call

from neo4j_graphrag.exceptions import SchemaValidationError
from neo4j_graphrag.experimental.pipeline.component import Component, DataModel


class SchemaProperty(BaseModel):
    """
    Represents a property on a node or relationship in the graph.
    """

    name: str
    # See https://neo4j.com/docs/cypher-manual/current/values-and-types/property-structural-constructed/#property-types
    type: Literal[
        "BOOLEAN",
        "DATE",
        "DURATION",
        "FLOAT",
        "INTEGER",
        "LIST",
        "LOCAL_DATETIME",
        "LOCAL_TIME",
        "POINT",
        "STRING",
        "ZONED_DATETIME",
        "ZONED_TIME",
    ]
    description: str = ""


class SchemaEntity(BaseModel):
    """
    Represents a possible node in the graph.
    """

    label: str
    description: str = ""
    properties: List[SchemaProperty] = []


class SchemaRelation(BaseModel):
    """
    Represents a possible relationship between nodes in the graph.
    """

    label: str
    description: str = ""
    properties: List[SchemaProperty] = []


class SchemaConfig(DataModel):
    """
    Represents possible relationships between entities and relations in the graph.
    """

    entities: Dict[str, Dict[str, Any]]
    relations: Dict[str, Dict[str, Any]]
    potential_schema: List[Tuple[str, str, str]]

    @model_validator(mode="before")
    def check_schema(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        entities = data.get("entities", {}).keys()
        relations = data.get("relations", {}).keys()
        potential_schema = data.get("potential_schema", [])

        for entity1, relation, entity2 in potential_schema:
            if entity1 not in entities:
                raise SchemaValidationError(
                    f"Entity '{entity1}' is not defined in the provided entities."
                )
            if relation not in relations:
                raise SchemaValidationError(
                    f"Relation '{relation}' is not defined in the provided relations."
                )
            if entity2 not in entities:
                raise SchemaValidationError(
                    f"Entity '{entity2}' is not defined in the provided entities."
                )

        return data


class SchemaBuilder(Component):
    """
    A builder class for constructing SchemaConfig objects from given entities,
    relations, and their interrelationships defined in a potential schema.

    Example:

    .. code-block:: python

        from neo4j_graphrag.experimental.components.schema import (
            SchemaBuilder,
            SchemaEntity,
            SchemaProperty,
            SchemaRelation,
        )
        from neo4j_graphrag.experimental.pipeline import Pipeline

        entities = [
            SchemaEntity(
                label="PERSON",
                description="An individual human being.",
                properties=[
                    SchemaProperty(
                        name="name", type="STRING", description="The name of the person"
                    )
                ],
            ),
            SchemaEntity(
                label="ORGANIZATION",
                description="A structured group of people with a common purpose.",
                properties=[
                    SchemaProperty(
                        name="name", type="STRING", description="The name of the organization"
                    )
                ],
            ),
        ]
        relations = [
            SchemaRelation(
                label="EMPLOYED_BY", description="Indicates employment relationship."
            ),
        ]
        potential_schema = [
            ("PERSON", "EMPLOYED_BY", "ORGANIZATION"),
        ]
        pipe = Pipeline()
        schema_builder = SchemaBuilder()
        pipe.add_component(schema_builder, "schema_builder")
        pipe_inputs = {
            "schema": {
                "entities": entities,
                "relations": relations,
                "potential_schema": potential_schema,
            },
            ...
        }
        pipe.run(pipe_inputs)
    """

    @staticmethod
    def create_schema_model(
        entities: List[SchemaEntity],
        relations: List[SchemaRelation],
        potential_schema: List[Tuple[str, str, str]],
    ) -> SchemaConfig:
        """
        Creates a SchemaConfig object from Lists of Entity and Relation objects
        and a Dictionary defining potential relationships.

        Args:
            entities (List[SchemaEntity]): List of Entity objects.
            relations (List[SchemaRelation]): List of Relation objects.
            potential_schema (Dict[str, List[str]]): Dictionary mapping entity names to Lists of relation names.

        Returns:
            SchemaConfig: A configured schema object.
        """
        entity_dict = {entity.label: entity.model_dump() for entity in entities}
        relation_dict = {
            relation.label: relation.model_dump() for relation in relations
        }

        try:
            return SchemaConfig(
                entities=entity_dict,
                relations=relation_dict,
                potential_schema=potential_schema,
            )
        except (ValidationError, SchemaValidationError) as e:
            raise SchemaValidationError(e)

    @validate_call
    async def run(
        self,
        entities: List[SchemaEntity],
        relations: List[SchemaRelation],
        potential_schema: List[Tuple[str, str, str]],
    ) -> SchemaConfig:
        """
        Asynchronously constructs and returns a SchemaConfig object.

        Args:
            entities (List[SchemaEntity]): List of Entity objects.
            relations (List[SchemaRelation]): List of Relation objects.
            potential_schema (Dict[str, List[str]]): Dictionary mapping entity names to Lists of relation names.

        Returns:
            SchemaConfig: A configured schema object, constructed asynchronously.
        """
        return self.create_schema_model(entities, relations, potential_schema)
