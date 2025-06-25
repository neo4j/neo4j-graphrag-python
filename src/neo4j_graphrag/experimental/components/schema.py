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
from typing import Any, Dict, List, Optional, Tuple, Sequence

import neo4j
from pydantic import (
    validate_call,
    ValidationError,
)

from neo4j_graphrag.exceptions import (
    SchemaValidationError,
    SchemaDatabaseConflictError,
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
    GraphSchema,
    SchemaConstraint,
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

    def _get_constraints_from_db(self) -> list[SchemaConstraint]:
        constraints = get_constraints(
            self.driver, database=self.neo4j_database, sanitize=False
        )
        return [
            SchemaConstraint.model_validate(c)
            for c in constraints
        ]

    def _apply_all_constraints_from_db(
        self,
        constraints: list[SchemaConstraint],
        entities: tuple[GraphEntityType, ...],
    ) -> list[GraphEntityType]:
        constrained_entity_types = []
        for entity_type in entities:
            new_entity_type = copy.deepcopy(entity_type)
            # find constraints related to this node type
            for constraint in constraints:
                if constraint.entity_type != entity_type.entity_type_name:
                    continue
                if constraint.label_or_type[0] != entity_type.label:
                    continue
                # now we can add the constraint to this node type
                self._apply_constraint_from_db(new_entity_type, constraint)
            constrained_entity_types.append(new_entity_type)
        return constrained_entity_types

    @staticmethod
    def _parse_property_type(property_type: str) -> list[Neo4jPropertyType]:
        prop_types = []
        if not property_type:
            return prop_types
        for prop_str in property_type.split("|"):
            p = prop_str.strip()
            try:
                prop = Neo4jPropertyType(p)
                prop_types.append(prop)
            except ValueError:
                pass
        return prop_types

    def _validate_constraint_compatibility(
        self, 
        entity_type: GraphEntityType, 
        constraint: SchemaConstraint
    ) -> None:
        """Validate that user schema is compatible with database constraint.
        
        Raises SchemaDatabaseConflictError if conflicts are detected.
        """
        # Rule 1: Missing Property Error
        missing_props = []
        for prop_name in constraint.properties:
            if entity_type.get_property_by_name(prop_name) is None:
                missing_props.append(prop_name)
        
        if missing_props:
            raise SchemaDatabaseConflictError(
                f"Database constraint {constraint.type} on {entity_type.label} "
                f"requires properties {missing_props} that are not defined in user schema. "
                f"Please add these properties to your {entity_type.label} definition or "
                f"remove the constraint from the database."
            )
        
        # Rule 2: Property Type Conflicts
        if constraint.type in (
            Neo4jConstraintTypeEnum.NODE_PROPERTY_TYPE,
            Neo4jConstraintTypeEnum.RELATIONSHIP_PROPERTY_TYPE
        ):
            self._validate_property_type_compatibility(entity_type, constraint)
        
        # Rule 3: Required Property Conflicts  
        if constraint.type in (
            Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            Neo4jConstraintTypeEnum.RELATIONSHIP_PROPERTY_EXISTENCE
        ):
            self._validate_required_property_compatibility(entity_type, constraint)

    def _validate_property_type_compatibility(
        self, 
        entity_type: GraphEntityType, 
        constraint: SchemaConstraint
    ) -> None:
        """Check if user property types are compatible with DB type constraints."""
        if not constraint.property_type:
            return
            
        for prop_name in constraint.properties:
            user_prop = entity_type.get_property_by_name(prop_name)
            if user_prop:
                user_types = user_prop.type if isinstance(user_prop.type, list) else [user_prop.type]
                db_allowed_types = constraint.property_type
                
                # Check if any user type is allowed by DB
                if not any(ut in db_allowed_types for ut in user_types):
                    raise SchemaDatabaseConflictError(
                        f"Property '{prop_name}' on {entity_type.label} has type {user_types} "
                        f"in user schema, but database constraint allows only {db_allowed_types}. "
                        f"Please update the property type or remove the database constraint."
                    )

    def _validate_required_property_compatibility(
        self, 
        entity_type: GraphEntityType, 
        constraint: SchemaConstraint
    ) -> None:
        """Check if user optional properties conflict with DB existence constraints.
        
        Only raises conflict if user explicitly set required=False when DB requires the property.
        If user left it as default (False), we can enhance it to required=True.
        """
        for prop_name in constraint.properties:
            user_prop = entity_type.get_property_by_name(prop_name)
            # For now, we'll be less strict and allow enhancement
            # In a future version, we could add a flag to track explicit vs default required=False
            # Currently, we'll only enhance the property to required=True without raising conflicts
            pass

    def _validate_missing_entity_types(
        self, 
        constraints: list[SchemaConstraint],
        user_node_labels: set[str],
        user_rel_types: set[str]
    ) -> None:
        """Check if DB has constraints on entity types not in user schema."""
        for constraint in constraints:
            if constraint.entity_type == "NODE":
                missing_labels = set(constraint.label_or_type) - user_node_labels
                if missing_labels:
                    raise SchemaDatabaseConflictError(
                        f"Database has constraints on node labels {missing_labels} "
                        f"that are not defined in user schema. Please add these node types "
                        f"or set additional_node_types=True."
                    )
            else:  # RELATIONSHIP
                missing_types = set(constraint.label_or_type) - user_rel_types
                if missing_types:
                    raise SchemaDatabaseConflictError(
                        f"Database has constraints on relationship types {missing_types} "
                        f"that are not defined in user schema. Please add these relationship types "
                        f"or set additional_relationship_types=True."
                    )

    def _validate_additional_properties_conflicts(
        self,
        entity_type: GraphEntityType,
        constraints: list[SchemaConstraint]
    ) -> None:
        """Check if additional_properties=False conflicts with DB constraints."""
        if entity_type.additional_properties:
            return  # No conflict if additional properties are allowed
        
        # Find all properties required by DB constraints for this entity
        required_by_db = set()
        for constraint in constraints:
            if (constraint.entity_type == entity_type.entity_type_name and 
                constraint.label_or_type[0] == entity_type.label):
                required_by_db.update(constraint.properties)
        
        # Check if any DB-required properties are missing from user schema
        user_properties = {prop.name for prop in entity_type.properties}
        missing_required = required_by_db - user_properties
        
        if missing_required:
            raise SchemaDatabaseConflictError(
                f"{entity_type.label} has additional_properties=False but database "
                f"constraints require properties {missing_required} not in user schema. "
                f"Please add these properties or set additional_properties=True."
            )

    def _apply_constraint_from_db(
        self, entity_type: GraphEntityType, constraint: SchemaConstraint,
    ) -> None:
        """Validate that user schema is compatible with database constraints.
        
        This method now focuses on validation and only applies safe enhancements.
        
        Args:
            entity_type: The entity type to validate and potentially enhance.
            constraint: The database constraint to validate against.
            
        Raises:
            SchemaDatabaseConflictError: If user schema conflicts with DB constraint.
        """
        # Step 1: Validate compatibility (raises errors for conflicts)
        self._validate_constraint_compatibility(entity_type, constraint)
        
        # Step 2: Apply only non-conflicting enhancements
        # (Only set required=True if user didn't explicitly set it to False)
        if constraint.type in (
            Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            Neo4jConstraintTypeEnum.RELATIONSHIP_PROPERTY_EXISTENCE,
        ):
            for prop_name in constraint.properties:
                prop = entity_type.get_property_by_name(prop_name)
                if prop and not prop.required:
                    # This would have been caught by validation if it was a conflict
                    prop.required = True

    def _create_schema_model(
        self,
        node_types: Sequence[EntityInputType],
        relationship_types: Optional[Sequence[RelationInputType]] = None,
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
            
        Raises:
            SchemaDatabaseConflictError: If user schema conflicts with database constraints.
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
        
        # Validate missing entity types before applying constraints
        user_node_labels = {node.label for node in schema.node_types}
        user_rel_types = {rel.label for rel in schema.relationship_types}
        
        # Check if schema allows additional types before validating missing entities
        additional_node_types = kwargs.get('additional_node_types', True)
        additional_relationship_types = kwargs.get('additional_relationship_types', True)
        
        if not additional_node_types or not additional_relationship_types:
            # Only validate missing entities if additional types are not allowed
            filtered_constraints = []
            for constraint in constraints:
                if constraint.entity_type == "NODE" and not additional_node_types:
                    filtered_constraints.append(constraint)
                elif constraint.entity_type == "RELATIONSHIP" and not additional_relationship_types:
                    filtered_constraints.append(constraint)
            
            if filtered_constraints:
                self._validate_missing_entity_types(
                    filtered_constraints, user_node_labels, user_rel_types
                )

        # Validate additional properties conflicts for each entity
        all_entities = list(schema.node_types) + list(schema.relationship_types)
        for entity in all_entities:
            self._validate_additional_properties_conflicts(entity, constraints)

        # apply constraints
        constrained_node_types = self._apply_all_constraints_from_db(
            constraints,
            schema.node_types,
        )
        constrained_relationship_types = self._apply_all_constraints_from_db(
            constraints,
            schema.relationship_types,
        )

        try:
            constrained_schema = GraphSchema.model_validate(
                dict(
                    node_types=constrained_node_types,
                    relationship_types=constrained_relationship_types,
                    patterns=schema.patterns,
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
            extracted_schema = json.loads(content)

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
