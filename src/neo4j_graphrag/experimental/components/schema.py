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

import copy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Sequence, Literal

import neo4j
from pydantic import (
    validate_call,
    ValidationError,
)

from neo4j_graphrag.exceptions import (
    SchemaValidationError,
    LLMGenerationError,
    SchemaExtractionError,
)
from neo4j_graphrag.experimental.pipeline.component import Component
from neo4j_graphrag.experimental.pipeline.types.schema import (
    EntityInputType,
    RelationInputType,
)
from neo4j_graphrag.generation import SchemaExtractionTemplate, PromptTemplate
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.experimental.components.types import (
    RelationshipType,
    GraphSchema,
    SchemaConstraint,
    ConstraintTypeEnum,
    Neo4jConstraintTypeEnum,
    GraphEntityType,
    Neo4jPropertyType,
)
from neo4j_graphrag.schema import get_constraints


class SchemaBuilder(Component):
    """
    A builder class for constructing GraphSchema objects from given entities,
    relations, and their interrelationships defined in a potential schema.

    Example:

    .. code-block:: python

        from neo4j_graphrag.experimental.components.schema import (
            SchemaBuilder,
            NodeType,
            PropertyType,
            RelationshipType,
        )
        from neo4j_graphrag.experimental.pipeline import Pipeline

        node_types = [
            NodeType(
                label="PERSON",
                description="An individual human being.",
                properties=[
                    PropertyType(
                        name="name", type="STRING", description="The name of the person"
                    )
                ],
            ),
            NodeType(
                label="ORGANIZATION",
                description="A structured group of people with a common purpose.",
                properties=[
                    PropertyType(
                        name="name", type="STRING", description="The name of the organization"
                    )
                ],
            ),
        ]
        relationship_types = [
            RelationshipType(
                label="EMPLOYED_BY", description="Indicates employment relationship."
            ),
        ]
        patterns = [
            ("PERSON", "EMPLOYED_BY", "ORGANIZATION"),
        ]
        pipe = Pipeline()
        schema_builder = SchemaBuilder()
        pipe.add_component(schema_builder, "schema_builder")
        pipe_inputs = {
            "schema": {
                "node_types": node_types,
                "relationship_types": relationship_types,
                "patterns": patterns,
            },
            ...
        }
        pipe.run(pipe_inputs)
    """

    def __init__(
        self, driver: neo4j.Driver, neo4j_database: Optional[str] = None
    ) -> None:
        self.driver = driver
        self.neo4j_database = neo4j_database

    def _get_constraints_from_db(self) -> list[dict[str, Any]]:
        constraints = get_constraints(
            self.driver, database=self.neo4j_database, sanitize=False
        )
        return constraints

    def _apply_all_constraints_from_db(
        self,
        node_or_relationship_type: Literal["NODE", "RELATIONSHIP"],
        constraints: list[dict[str, Any]],
        entities: tuple[GraphEntityType, ...],
    ) -> list[GraphEntityType]:
        constrained_entity_types = []
        for entity_type in entities:
            new_entity_type = copy.deepcopy(entity_type)
            # find constraints related to this node type
            for constraint in constraints:
                if constraint["entityType"] != node_or_relationship_type:
                    continue
                if constraint["labelsOrTypes"][0] != entity_type.label:
                    continue
                # now we can add the constraint to this node type
                self._apply_constraint_from_db(new_entity_type, constraint)
            constrained_entity_types.append(new_entity_type)
        return constrained_entity_types

    @staticmethod
    def _parse_property_type(property_type: str) -> Neo4jPropertyType | None:
        if not property_type:
            return None
        prop = None
        for prop_str in property_type.split("|"):
            p = prop_str.strip()
            try:
                prop = Neo4jPropertyType(p)
            except ValueError:
                pass
        return prop

    def _apply_constraint_from_db(
        self, entity_type: GraphEntityType, constraint: dict[str, Any]
    ) -> None:
        neo4j_constraint_type = Neo4jConstraintTypeEnum(constraint["type"])
        # TODO: detect potential conflict and raise ValueError if any
        # existing_schema_constraints_on_property = node_type.get_constraints_on_properties(constraint["properties"])
        constraint_properties = constraint["properties"]
        for p in constraint_properties:
            if entity_type.get_property_by_name(p) is None:
                raise ValueError(
                    f"Can not add constraint {constraint} on non existing property"
                )
        constraint_type = neo4j_constraint_type.to_constraint_type()
        entity_type.constraints.append(
            SchemaConstraint(
                type=constraint_type,
                properties=constraint["properties"],
                property_type=self._parse_property_type(constraint["propertyType"]),
                name=constraint["name"],
            )
        )
        # if property required constraint, make sure the flag is set properly on
        # the PropertyType
        if constraint_type == ConstraintTypeEnum.PROPERTY_EXISTENCE:
            prop = entity_type.get_property_by_name(constraint["properties"][0])
            if prop:
                prop.required = True
        return None

    def _create_schema_model(
        self,
        node_types: Sequence[EntityInputType],
        relationship_types: Optional[Sequence[RelationshipType]] = None,
        patterns: Optional[Sequence[Tuple[str, str, str]]] = None,
        **kwargs: Any,
    ) -> GraphSchema:
        """
        Creates a GraphSchema object from Lists of Entity and Relation objects
        and a Dictionary defining potential relationships.

        Args:
            node_types (Sequence[NodeType]): List or tuple of NodeType objects.
            relationship_types (Optional[Sequence[RelationshipType]]): List or tuple of RelationshipType objects.
            patterns (Optional[Sequence[Tuple[str, str, str]]]): List or tuples of triplets: (source_entity_label, relation_label, target_entity_label).
            kwargs: other arguments passed to GraphSchema validator.

        Returns:
            GraphSchema: A configured schema object.
        """
        try:
            schema = GraphSchema.model_validate(
                dict(
                    node_types=node_types,
                    relationship_types=relationship_types or (),
                    patterns=patterns or (),
                    **kwargs,
                )
            )
        except ValidationError as e:
            raise SchemaValidationError() from e

        constraints = self._get_constraints_from_db()
        # apply constraints
        constrained_node_types = self._apply_all_constraints_from_db(
            "NODE",
            constraints,
            schema.node_types,
        )
        constrained_relationship_types = self._apply_all_constraints_from_db(
            "RELATIONSHIP",
            constraints,
            schema.relationship_types,
        )

        try:
            constrained_schema = GraphSchema.model_validate(
                dict(
                    node_types=constrained_node_types,
                    relationship_types=constrained_relationship_types,
                    patterns=patterns,
                    **kwargs,
                )
            )
        except ValidationError as e:
            raise SchemaValidationError(
                "Error when applying constraints from database"
            ) from e
        return constrained_schema

    @validate_call
    async def run(
        self,
        node_types: Sequence[EntityInputType],
        relationship_types: Optional[Sequence[RelationInputType]] = None,
        patterns: Optional[Sequence[Tuple[str, str, str]]] = None,
        **kwargs: Any,
    ) -> GraphSchema:
        """
        Asynchronously constructs and returns a GraphSchema object.

        Args:
            node_types (Sequence[NodeType]): Sequence of NodeType objects.
            relationship_types (Sequence[RelationshipType]): Sequence of RelationshipType objects.
            patterns (Optional[Sequence[Tuple[str, str, str]]]): Sequence of triplets: (source_entity_label, relation_label, target_entity_label).

        Returns:
            GraphSchema: A configured schema object, constructed asynchronously.
        """
        return self._create_schema_model(
            node_types,
            relationship_types,
            patterns,
            **kwargs,
        )


class SchemaFromTextExtractor(Component):
    """
    A component for constructing GraphSchema objects from the output of an LLM after
    automatic schema extraction from text.
    """

    def __init__(
        self,
        driver: neo4j.Driver,
        llm: LLMInterface,
        prompt_template: Optional[PromptTemplate] = None,
        llm_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.driver = driver
        self._llm: LLMInterface = llm
        self._prompt_template: PromptTemplate = (
            prompt_template or SchemaExtractionTemplate()
        )
        self._llm_params: dict[str, Any] = llm_params or {}

    @validate_call
    async def run(self, text: str, examples: str = "", **kwargs: Any) -> GraphSchema:
        """
        Asynchronously extracts the schema from text and returns a GraphSchema object.

        Args:
            text (str): the text from which the schema will be inferred.
            examples (str): examples to guide schema extraction.
        Returns:
            GraphSchema: A configured schema object, extracted automatically and
            constructed asynchronously.
        """
        prompt: str = self._prompt_template.format(text=text, examples=examples)

        try:
            response = await self._llm.ainvoke(prompt, **self._llm_params)
            content: str = response.content
        except LLMGenerationError as e:
            # Re-raise the LLMGenerationError
            raise LLMGenerationError("Failed to generate schema from text") from e

        try:
            extracted_schema: Dict[str, Any] = json.loads(content)

            # handle dictionary
            if isinstance(extracted_schema, dict):
                pass  # Keep as is
            # handle list
            elif isinstance(extracted_schema, list):
                if len(extracted_schema) > 0 and isinstance(extracted_schema[0], dict):
                    extracted_schema = extracted_schema[0]
                elif len(extracted_schema) == 0:
                    logging.warning(
                        "LLM returned an empty list for schema. Falling back to empty schema."
                    )
                    extracted_schema = {}
                else:
                    raise SchemaExtractionError(
                        f"Expected a dictionary or list of dictionaries, but got list containing: {type(extracted_schema[0])}"
                    )
            # any other types
            else:
                raise SchemaExtractionError(
                    f"Unexpected schema format returned from LLM: {type(extracted_schema)}. Expected a dictionary or list of dictionaries."
                )
        except json.JSONDecodeError as exc:
            raise SchemaExtractionError("LLM response is not valid JSON.") from exc

        extracted_node_types: List[Dict[str, Any]] = (
            extracted_schema.get("node_types") or []
        )
        extracted_relationship_types: Optional[List[Dict[str, Any]]] = (
            extracted_schema.get("relationship_types")
        )
        extracted_patterns: Optional[List[Tuple[str, str, str]]] = extracted_schema.get(
            "patterns"
        )

        return GraphSchema.model_validate(
            {
                "node_types": extracted_node_types,
                "relationship_types": extracted_relationship_types,
                "patterns": extracted_patterns,
            }
        )
