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
    NodeType,
    PropertyType,
    RelationshipType,
)
from neo4j_graphrag.schema import get_constraints


class ConstraintProcessor:
    """
    Base class for processing database constraints against schemas.
    
    Provides shared logic for both validation and enhancement modes when working
    with Neo4j database constraints. This class handles the core constraint processing
    functionality used by both SchemaBuilder and SchemaFromTextExtractor.
    
    The constraint processor can operate in two modes:
    - **Validation Mode**: Validates schemas against constraints and raises errors for conflicts
    - **Enhancement Mode**: Automatically modifies schemas to resolve constraint conflicts
    
    Args:
        driver: Neo4j driver instance for database access
        neo4j_database: Optional Neo4j database name. If None, uses default database.
        
    Attributes:
        driver: The Neo4j driver instance
        neo4j_database: The Neo4j database name
    """
    
    def __init__(self, driver: neo4j.Driver, neo4j_database: Optional[str] = None):
        """
        Initialize the ConstraintProcessor.
        
        Args:
            driver: Neo4j driver instance for database access
            neo4j_database: Optional Neo4j database name. If None, uses default database.
        """
        self.driver = driver
        self.neo4j_database = neo4j_database

    def _get_constraints_from_db(self) -> list[SchemaConstraint]:
        """
        Retrieve all constraints from the Neo4j database.
        
        Returns:
            List of SchemaConstraint objects representing database constraints.
            
        Note:
            This method uses the get_constraints utility function to fetch constraints
            from the database and converts them to SchemaConstraint objects.
        """
        constraints = get_constraints(
            self.driver, database=self.neo4j_database, sanitize=False
        )
        return [
            SchemaConstraint.model_validate(c)
            for c in constraints
        ]

    @staticmethod
    def _parse_property_type(property_type: str) -> list[Neo4jPropertyType]:
        """
        Parse a property type string into a list of Neo4jPropertyType enums.
        
        Args:
            property_type: String representation of property types (e.g., "STRING|INTEGER")
            
        Returns:
            List of Neo4jPropertyType enums. Empty list if property_type is empty or invalid.
            
        Note:
            This method handles pipe-separated type strings and ignores invalid types.
        """
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

    def _infer_property_type_from_constraint(
        self, 
        constraint: SchemaConstraint
    ) -> Neo4jPropertyType:
        """
        Infer the best property type from a database constraint.
        
        Args:
            constraint: The database constraint to analyze
            
        Returns:
            Neo4jPropertyType: The inferred property type. Defaults to STRING if no
            specific type is defined in the constraint.
            
        Note:
            For type constraints, returns the first allowed type. For existence
            constraints without type information, defaults to STRING.
        """
        if constraint.property_type:
            # Use the first type from the constraint
            return constraint.property_type[0]
        else:
            # Default to STRING for existence constraints
            return Neo4jPropertyType.STRING

    def _check_missing_properties(
        self, 
        entity_type: GraphEntityType, 
        constraint: SchemaConstraint
    ) -> list[str]:
        """
        Check for properties required by constraint but missing from entity.
        
        Args:
            entity_type: The entity type to check
            constraint: The database constraint to validate against
            
        Returns:
            List of property names that are required by the constraint but missing
            from the entity definition.
        """
        missing_props = []
        for prop_name in constraint.properties:
            if entity_type.get_property_by_name(prop_name) is None:
                missing_props.append(prop_name)
        return missing_props

    def _check_property_type_compatibility(
        self, 
        entity_type: GraphEntityType, 
        constraint: SchemaConstraint
    ) -> list[tuple[str, list[Neo4jPropertyType], list[Neo4jPropertyType]]]:
        """
        Check for property type conflicts between entity and constraint.
        
        Args:
            entity_type: The entity type to check
            constraint: The database constraint to validate against
            
        Returns:
            List of tuples containing (property_name, user_types, database_allowed_types)
            for each property that has a type conflict.
            
        Note:
            Only checks type constraints. Returns empty list for constraints without
            type information.
        """
        conflicts = []
        if not constraint.property_type:
            return conflicts
            
        for prop_name in constraint.properties:
            user_prop = entity_type.get_property_by_name(prop_name)
            if user_prop:
                user_types = user_prop.type if isinstance(user_prop.type, list) else [user_prop.type]
                db_allowed_types = constraint.property_type
                
                # Check if any user type is allowed by DB
                if not any(ut in db_allowed_types for ut in user_types):
                    conflicts.append((prop_name, user_types, db_allowed_types))
        return conflicts

    def _check_missing_entity_types(
        self, 
        constraints: list[SchemaConstraint],
        user_node_labels: set[str],
        user_rel_types: set[str],
        additional_node_types: bool,
        additional_relationship_types: bool
    ) -> tuple[set[str], set[str]]:
        """
        Check for entity types required by constraints but missing from schema.
        
        Args:
            constraints: List of database constraints
            user_node_labels: Set of node labels defined in user schema
            user_rel_types: Set of relationship types defined in user schema
            additional_node_types: Whether additional node types are allowed
            additional_relationship_types: Whether additional relationship types are allowed
            
        Returns:
            Tuple of (missing_node_labels, missing_relationship_types) that are
            required by constraints but not defined in the schema and not allowed
            by the additional_*_types flags.
        """
        missing_node_labels = set()
        missing_rel_types = set()
        
        for constraint in constraints:
            if constraint.entity_type == "NODE" and not additional_node_types:
                missing_labels = set(constraint.label_or_type) - user_node_labels
                missing_node_labels.update(missing_labels)
            elif constraint.entity_type == "RELATIONSHIP" and not additional_relationship_types:
                missing_types = set(constraint.label_or_type) - user_rel_types
                missing_rel_types.update(missing_types)
        
        return missing_node_labels, missing_rel_types

    def _check_additional_properties_conflicts(
        self,
        entity_type: GraphEntityType,
        constraints: list[SchemaConstraint]
    ) -> set[str]:
        """
        Check for conflicts when entity has additional_properties=False.
        
        Args:
            entity_type: The entity type to check
            constraints: List of database constraints
            
        Returns:
            Set of property names that are required by database constraints but
            missing from the entity schema when additional_properties=False.
            
        Note:
            Returns empty set if additional_properties=True, as no conflict exists
            in that case.
        """
        if entity_type.additional_properties:
            return set()  # No conflict if additional properties are allowed
        
        # Find all properties required by DB constraints for this entity
        required_by_db = set()
        for constraint in constraints:
            if (constraint.entity_type == entity_type.entity_type_name and 
                constraint.label_or_type[0] == entity_type.label):
                required_by_db.update(constraint.properties)
        
        # Check if any DB-required properties are missing from user schema
        user_properties = {prop.name for prop in entity_type.properties}
        missing_required = required_by_db - user_properties
        
        return missing_required

    def _process_constraints_against_schema(
        self,
        schema: GraphSchema,
        mode: str = "validate",  # "validate" or "enhance"
        **kwargs: Any
    ) -> GraphSchema:
        """
        Process database constraints against schema in either validation or enhancement mode.
        
        This is the main constraint processing method that handles both validation and
        enhancement modes. It coordinates all constraint checking and resolution logic.
        
        Args:
            schema: The schema to process
            mode: Processing mode - "validate" to raise errors, "enhance" to modify schema
            **kwargs: Additional configuration parameters including:
                - additional_node_types (bool): Whether to allow additional node types
                - additional_relationship_types (bool): Whether to allow additional relationship types
                
        Returns:
            GraphSchema: The processed schema. In validation mode, returns the original
            schema with safe enhancements (like required=True). In enhancement mode,
            returns a modified schema with added properties and entity types.
            
        Raises:
            SchemaDatabaseConflictError: If mode="validate" and conflicts are found
            ValueError: If mode is not "validate" or "enhance"
            
        Note:
            This method is the core of the constraint processing system and is used
            by both SchemaBuilder and SchemaFromTextExtractor.
        """
        constraints = self._get_constraints_from_db()
        if not constraints:
            return schema  # No constraints to process
        
        # Get configuration flags
        additional_node_types = kwargs.get('additional_node_types', True)
        additional_relationship_types = kwargs.get('additional_relationship_types', True)
        
        # Check for conflicts
        user_node_labels = {node.label for node in schema.node_types}
        user_rel_types = {rel.label for rel in schema.relationship_types}
        
        # 1. Check missing entity types
        missing_node_labels, missing_rel_types = self._check_missing_entity_types(
            constraints, user_node_labels, user_rel_types, 
            additional_node_types, additional_relationship_types
        )
        
        if mode == "validate":
            # Validation mode: raise errors for conflicts
            self._validate_missing_entity_types(missing_node_labels, missing_rel_types)
            self._validate_entity_constraints(schema.node_types + schema.relationship_types, constraints)
            
            # Apply only safe enhancements (required=True)
            enhanced_entities = self._apply_safe_enhancements(
                list(schema.node_types) + list(schema.relationship_types), constraints
            )
            
            return self._rebuild_schema_with_entities(schema, enhanced_entities, **kwargs)
            
        elif mode == "enhance":
            # Enhancement mode: modify schema to resolve conflicts
            return self._enhance_schema_with_constraints(schema, constraints, **kwargs)
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'validate' or 'enhance'")

    def _validate_missing_entity_types(self, missing_node_labels: set[str], missing_rel_types: set[str]) -> None:
        """
        Raise errors for missing entity types in validation mode.
        
        Args:
            missing_node_labels: Set of node labels required by constraints but missing from schema
            missing_rel_types: Set of relationship types required by constraints but missing from schema
            
        Raises:
            SchemaDatabaseConflictError: If any missing entity types are found
        """
        if missing_node_labels:
            raise SchemaDatabaseConflictError(
                f"Database has constraints on node labels {missing_node_labels} "
                f"that are not defined in user schema. Please add these node types "
                f"or set additional_node_types=True."
            )
        if missing_rel_types:
            raise SchemaDatabaseConflictError(
                f"Database has constraints on relationship types {missing_rel_types} "
                f"that are not defined in user schema. Please add these relationship types "
                f"or set additional_relationship_types=True."
            )

    def _validate_entity_constraints(
        self, 
        entities: list[GraphEntityType], 
        constraints: list[SchemaConstraint]
    ) -> None:
        """
        Validate all entity constraints in validation mode.
        
        Args:
            entities: List of entities (nodes and relationships) to validate
            constraints: List of database constraints to validate against
            
        Raises:
            SchemaDatabaseConflictError: If any constraint conflicts are found
            
        Note:
            This method performs comprehensive validation including missing properties,
            property type conflicts, and additional_properties conflicts.
        """
        for entity in entities:
            relevant_constraints = [
                c for c in constraints 
                if (c.entity_type == entity.entity_type_name and 
                    entity.label in c.label_or_type)
            ]
            
            # Check additional properties conflicts first (more specific error)
            missing_required = self._check_additional_properties_conflicts(entity, relevant_constraints)
            if missing_required:
                raise SchemaDatabaseConflictError(
                    f"{entity.label} has additional_properties=False but database "
                    f"constraints require properties {missing_required} not in user schema. "
                    f"Please add these properties or set additional_properties=True."
                )
            
            for constraint in relevant_constraints:
                # Check missing properties
                missing_props = self._check_missing_properties(entity, constraint)
                if missing_props:
                    raise SchemaDatabaseConflictError(
                        f"Database constraint {constraint.type} on {entity.label} "
                        f"requires properties {missing_props} that are not defined in user schema. "
                        f"Please add these properties to your {entity.label} definition or "
                        f"remove the constraint from the database."
                    )
                
                # Check property type conflicts
                type_conflicts = self._check_property_type_compatibility(entity, constraint)
                if type_conflicts:
                    for prop_name, user_types, db_types in type_conflicts:
                        raise SchemaDatabaseConflictError(
                            f"Property '{prop_name}' on {entity.label} has type {user_types} "
                            f"in user schema, but database constraint allows only {db_types}. "
                            f"Please update the property type or remove the database constraint."
                        )

    def _apply_safe_enhancements(
        self,
        entities: list[GraphEntityType],
        constraints: list[SchemaConstraint]
    ) -> list[GraphEntityType]:
        """
        Apply only safe enhancements that don't add new properties.
        
        Safe enhancements include setting required=True for properties that already
        exist in the schema but are required by database existence constraints.
        
        Args:
            entities: List of entities to enhance
            constraints: List of database constraints
            
        Returns:
            List of enhanced entities with safe modifications applied
            
        Note:
            This method is used in validation mode to apply non-conflicting
            enhancements while preserving the user's original schema structure.
        """
        enhanced_entities = [copy.deepcopy(entity) for entity in entities]
        
        for entity in enhanced_entities:
            relevant_constraints = [
                c for c in constraints 
                if (c.entity_type == entity.entity_type_name and 
                    entity.label in c.label_or_type)
            ]
            
            for constraint in relevant_constraints:
                if constraint.type in (
                    Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
                    Neo4jConstraintTypeEnum.RELATIONSHIP_PROPERTY_EXISTENCE,
                ):
                    for prop_name in constraint.properties:
                        prop = entity.get_property_by_name(prop_name)
                        if prop and not prop.required:
                            prop.required = True
        
        return enhanced_entities

    def _enhance_schema_with_constraints(
        self, 
        schema: GraphSchema, 
        constraints: list[SchemaConstraint],
        **kwargs: Any
    ) -> GraphSchema:
        """
        Enhance schema by adding missing entities and properties in enhancement mode.
        
        This method performs comprehensive schema enhancement including:
        - Adding missing entity types required by constraints
        - Adding missing properties to existing entities
        - Updating property types to match constraints
        - Setting required flags for existence constraints
        
        Args:
            schema: The original schema to enhance
            constraints: List of database constraints to apply
            **kwargs: Configuration parameters including additional_*_types flags
            
        Returns:
            GraphSchema: Enhanced schema with all constraint conflicts resolved.
            If enhancement fails, returns the original schema with a warning logged.
            
        Note:
            This method is used by SchemaFromTextExtractor and SchemaBuilder in
            enhancement mode to automatically resolve constraint conflicts.
        """
        # Create mutable copies of entities
        enhanced_node_types = [copy.deepcopy(node) for node in schema.node_types]
        enhanced_relationship_types = [copy.deepcopy(rel) for rel in schema.relationship_types]
        
        # Get configuration flags
        additional_node_types = kwargs.get('additional_node_types', True)
        additional_relationship_types = kwargs.get('additional_relationship_types', True)
        
        # Step 1: Add missing entity types required by constraints
        enhanced_node_types = self._add_missing_entity_types(
            enhanced_node_types, constraints, "NODE", additional_node_types
        )
        enhanced_relationship_types = self._add_missing_entity_types(
            enhanced_relationship_types, constraints, "RELATIONSHIP", additional_relationship_types
        )
        
        # Step 2: Enhance existing entities with constraint requirements
        all_enhanced_entities = enhanced_node_types + enhanced_relationship_types
        for entity in all_enhanced_entities:
            self._enhance_entity_with_constraints(entity, constraints)
        
        # Step 3: Create enhanced schema
        try:
            enhanced_schema = GraphSchema.model_validate(
                dict(
                    node_types=enhanced_node_types,
                    relationship_types=enhanced_relationship_types,
                    patterns=schema.patterns,
                    **kwargs,
                )
            )
        except ValidationError as e:
            # If enhancement fails, log warning and return original schema
            logging.warning(f"Failed to enhance schema with constraints: {e}")
            return schema
            
        return enhanced_schema

    def _add_missing_entity_types(
        self,
        existing_entities: list[GraphEntityType],
        constraints: list[SchemaConstraint],
        entity_type: str,
        allow_additional: bool
    ) -> list[GraphEntityType]:
        """
        Add entity types that are required by database constraints but missing from schema.
        
        Creates minimal entity definitions for missing entity types that are referenced
        by database constraints. Each created entity includes all properties required
        by the relevant constraints.
        
        Args:
            existing_entities: Current list of entities in the schema
            constraints: Database constraints to analyze
            entity_type: Type of entities to add - "NODE" or "RELATIONSHIP"
            allow_additional: Whether to add missing types (respects additional_*_types flags)
            
        Returns:
            Enhanced list of entities with missing types added. If allow_additional is False,
            returns the original list unchanged.
            
        Note:
            Created entities have additional_properties=True to allow flexibility and
            include descriptive text indicating they were added due to constraints.
        """
        if not allow_additional:
            return existing_entities
            
        existing_labels = {entity.label for entity in existing_entities}
        
        # Find labels referenced by constraints but not in schema
        required_labels = set()
        for constraint in constraints:
            if constraint.entity_type == entity_type:
                required_labels.update(constraint.label_or_type)
        
        missing_labels = required_labels - existing_labels
        
        # Create minimal entity definitions for missing labels
        enhanced_entities = list(existing_entities)
        for label in missing_labels:
            # Create a basic entity with properties required by constraints
            required_properties = []
            for constraint in constraints:
                if (constraint.entity_type == entity_type and 
                    label in constraint.label_or_type):
                    for prop_name in constraint.properties:
                        # Add property if not already added
                        if not any(p.name == prop_name for p in required_properties):
                            prop_type = self._infer_property_type_from_constraint(constraint)
                            required_properties.append(
                                PropertyType(
                                    name=prop_name,
                                    type=prop_type,
                                    required=constraint.type in (
                                        Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
                                        Neo4jConstraintTypeEnum.RELATIONSHIP_PROPERTY_EXISTENCE
                                    ),
                                    description=f"Property required by database constraint"
                                )
                            )
            
            # Create the entity
            if entity_type == "NODE":
                new_entity = NodeType(
                    label=label,
                    description=f"Node type added to match database constraints",
                    properties=required_properties,
                    additional_properties=True  # Allow flexibility
                )
            else:  # RELATIONSHIP
                new_entity = RelationshipType(
                    label=label,
                    description=f"Relationship type added to match database constraints", 
                    properties=required_properties,
                    additional_properties=True
                )
            
            enhanced_entities.append(new_entity)
            logging.info(f"Added missing {entity_type.lower()} type '{label}' to match database constraints")
        
        return enhanced_entities

    def _enhance_entity_with_constraints(
        self,
        entity: GraphEntityType,
        constraints: list[SchemaConstraint]
    ) -> None:
        """
        Enhance an entity by adding missing properties and updating existing ones.
        
        This method modifies the entity in-place to match database constraint requirements.
        It adds missing properties (if additional_properties=True) and enhances existing
        properties with constraint information.
        
        Args:
            entity: The entity to enhance (modified in-place)
            constraints: Database constraints to apply
            
        Note:
            Only adds properties if entity.additional_properties is True. Always
            enhances existing properties regardless of additional_properties setting.
        """
        relevant_constraints = [
            c for c in constraints 
            if (c.entity_type == entity.entity_type_name and 
                entity.label in c.label_or_type)
        ]
        
        if not relevant_constraints:
            return
            
        for constraint in relevant_constraints:
            for prop_name in constraint.properties:
                existing_prop = entity.get_property_by_name(prop_name)
                
                if existing_prop:
                    # Enhance existing property
                    self._enhance_property_with_constraint(existing_prop, constraint)
                else:
                    # Add missing property
                    if entity.additional_properties:
                        # Only add if additional properties are allowed
                        prop_type = self._infer_property_type_from_constraint(constraint)
                        new_prop = PropertyType(
                            name=prop_name,
                            type=prop_type,
                            required=constraint.type in (
                                Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
                                Neo4jConstraintTypeEnum.RELATIONSHIP_PROPERTY_EXISTENCE
                            ),
                            description=f"Property added to match database constraint"
                        )
                        entity.properties.append(new_prop)
                        logging.info(f"Added missing property '{prop_name}' to {entity.label}")

    def _enhance_property_with_constraint(
        self,
        prop: PropertyType,
        constraint: SchemaConstraint
    ) -> None:
        """
        Enhance a property to match database constraint requirements.
        
        This method modifies the property in-place to align with database constraints.
        It can set required=True for existence constraints and update property types
        for type constraints.
        
        Args:
            prop: The property to enhance (modified in-place)
            constraint: The constraint to apply
            
        Note:
            For type constraints, only updates the property type if the current type
            is not compatible with the constraint. Uses the first allowed type from
            the constraint.
        """
        # Set required=True for existence constraints
        if constraint.type in (
            Neo4jConstraintTypeEnum.NODE_PROPERTY_EXISTENCE,
            Neo4jConstraintTypeEnum.RELATIONSHIP_PROPERTY_EXISTENCE
        ):
            if not prop.required:
                prop.required = True
                logging.info(f"Enhanced property '{prop.name}' to required=True due to database constraint")
        
        # Update property type for type constraints
        if (constraint.type in (
            Neo4jConstraintTypeEnum.NODE_PROPERTY_TYPE,
            Neo4jConstraintTypeEnum.RELATIONSHIP_PROPERTY_TYPE
        ) and constraint.property_type):
            current_types = prop.type if isinstance(prop.type, list) else [prop.type]
            
            # Check if current type is compatible with constraint
            if not any(ct in constraint.property_type for ct in current_types):
                # Use the first allowed type from constraint
                new_type = constraint.property_type[0]
                prop.type = new_type
                logging.info(f"Enhanced property '{prop.name}' type to {new_type} due to database constraint")

    def _rebuild_schema_with_entities(
        self,
        original_schema: GraphSchema,
        enhanced_entities: list[GraphEntityType],
        **kwargs: Any
    ) -> GraphSchema:
        """
        Rebuild a GraphSchema with enhanced entities.
        
        Args:
            original_schema: The original schema to preserve patterns and other settings
            enhanced_entities: List of enhanced entities (both nodes and relationships)
            **kwargs: Additional schema configuration parameters
            
        Returns:
            GraphSchema: New schema with enhanced entities
            
        Raises:
            SchemaValidationError: If the enhanced schema fails validation
        """
        # Split entities back into nodes and relationships
        enhanced_nodes = [e for e in enhanced_entities if e.entity_type_name == "NODE"]
        enhanced_rels = [e for e in enhanced_entities if e.entity_type_name == "RELATIONSHIP"]
        
        try:
            return GraphSchema.model_validate(
                dict(
                    node_types=enhanced_nodes,
                    relationship_types=enhanced_rels,
                    patterns=original_schema.patterns,
                    **kwargs,
                )
            )
        except ValidationError as e:
            raise SchemaValidationError("Error when applying constraints from database") from e


class SchemaBuilder(Component, ConstraintProcessor):
    """
    A builder class for constructing GraphSchema objects from manually defined entities,
    relations, and their interrelationships.

    SchemaBuilder supports two modes of operation:
    
    - **Validation Mode** (default): Validates user schemas against database constraints 
      and raises SchemaDatabaseConflictError for conflicts. Use when you want explicit 
      control over your schema.
    - **Enhancement Mode**: Automatically modifies schemas to resolve conflicts with 
      database constraints. Use during development or when you want automatic compatibility.

    Args:
        driver: Neo4j driver instance for database access
        neo4j_database: Optional Neo4j database name. If None, uses default database.
        enhancement_mode: If True, enhances schema instead of raising errors.
                         If False (default), validates schema and raises errors for conflicts.

    Example:
        .. code-block:: python

            from neo4j_graphrag.experimental.components.schema import (
                SchemaBuilder,
                NodeType,
                PropertyType,
                RelationshipType,
            )

            # Validation mode (default) - raises errors for conflicts
            schema_builder = SchemaBuilder(driver, enhancement_mode=False)
            
            # Enhancement mode - automatically fixes conflicts  
            schema_builder = SchemaBuilder(driver, enhancement_mode=True)

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
            
            # This will validate/enhance the schema against database constraints
            schema = await schema_builder.run(
                node_types=node_types,
                relationship_types=relationship_types,
                patterns=patterns,
            )
    """

    def __init__(
        self, 
        driver: neo4j.Driver, 
        neo4j_database: Optional[str] = None,
        enhancement_mode: bool = False
    ) -> None:
        """
        Initialize SchemaBuilder with constraint processing capabilities.
        
        Args:
            driver: Neo4j driver instance for database access
            neo4j_database: Optional Neo4j database name. If None, uses default database.
            enhancement_mode: If True, enhances schema instead of raising errors.
                             If False (default), validates schema and raises errors for conflicts.
        """
        super().__init__(driver, neo4j_database)
        self.enhancement_mode = enhancement_mode

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

    def _validate_constraint_compatibility(
        self, 
        entity_type: GraphEntityType, 
        constraint: SchemaConstraint
    ) -> None:
        """Legacy method for backward compatibility. Uses shared logic."""
        missing_props = self._check_missing_properties(entity_type, constraint)
        if missing_props:
            raise SchemaDatabaseConflictError(
                f"Database constraint {constraint.type} on {entity_type.label} "
                f"requires properties {missing_props} that are not defined in user schema. "
                f"Please add these properties to your {entity_type.label} definition or "
                f"remove the constraint from the database."
            )
        
        # Check property type conflicts
        if constraint.type in (
            Neo4jConstraintTypeEnum.NODE_PROPERTY_TYPE,
            Neo4jConstraintTypeEnum.RELATIONSHIP_PROPERTY_TYPE
        ):
            type_conflicts = self._check_property_type_compatibility(entity_type, constraint)
            if type_conflicts:
                for prop_name, user_types, db_types in type_conflicts:
                    raise SchemaDatabaseConflictError(
                        f"Property '{prop_name}' on {entity_type.label} has type {user_types} "
                        f"in user schema, but database constraint allows only {db_types}. "
                        f"Please update the property type or remove the database constraint."
                    )

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
            SchemaDatabaseConflictError: If enhancement_mode=False and conflicts found.
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

        # Use shared constraint processing logic
        mode = "enhance" if self.enhancement_mode else "validate"
        return self._process_constraints_against_schema(schema, mode=mode, **kwargs)

    @validate_call
    async def run(
        self,
        node_types: Sequence[EntityInputType],
        relationship_types: Optional[Sequence[RelationInputType]] = None,
        patterns: Optional[Sequence[Tuple[str, str, str]]] = None,
        **kwargs: Any,
    ) -> GraphSchema:
        """
        Asynchronously constructs and returns a GraphSchema object with constraint processing.

        This method creates a schema from the provided entities and processes it against
        database constraints according to the configured mode (validation or enhancement).

        Args:
            node_types: Sequence of NodeType objects defining the node types in the schema
            relationship_types: Optional sequence of RelationshipType objects defining 
                              the relationship types in the schema
            patterns: Optional sequence of triplets (source_entity_label, relation_label, 
                     target_entity_label) defining allowed relationship patterns
            **kwargs: Additional configuration parameters including:
                - additional_node_types (bool): Whether to allow additional node types
                - additional_relationship_types (bool): Whether to allow additional relationship types
                - additional_properties (bool): Whether entities can have additional properties

        Returns:
            GraphSchema: A configured schema object. In validation mode, returns the schema
            with safe enhancements (like required=True). In enhancement mode, returns a
            schema modified to resolve all constraint conflicts.
            
        Raises:
            SchemaDatabaseConflictError: If enhancement_mode=False and conflicts are found
            SchemaValidationError: If the provided entity definitions are invalid
            
        Note:
            The schema is automatically processed against database constraints using the
            shared constraint processing logic from ConstraintProcessor.
        """
        return self._create_schema_model(
            node_types,
            relationship_types,
            patterns,
            **kwargs,
        )


class SchemaFromTextExtractor(Component, ConstraintProcessor):
    """
    A component for automatically extracting GraphSchema objects from text using LLMs.
    
    This component uses a Large Language Model to analyze text and automatically extract
    entity types, relationship types, and their properties. The extracted schema is then
    automatically enhanced to match database constraints.
    
    SchemaFromTextExtractor always operates in enhancement mode - it never raises errors
    for constraint conflicts. Instead, it automatically modifies the LLM-generated schema
    to ensure database compatibility.

    Args:
        driver: Neo4j driver instance for database access
        llm: LLM instance implementing LLMInterface for schema extraction
        prompt_template: Optional custom prompt template for schema extraction.
                        Defaults to SchemaExtractionTemplate.
        llm_params: Optional dictionary of additional parameters to pass to the LLM
        neo4j_database: Optional Neo4j database name. If None, uses default database.

    Example:
        .. code-block:: python

            from neo4j_graphrag.experimental.components.schema import SchemaFromTextExtractor
            from neo4j_graphrag.llm import OpenAILLM

            # Create the schema extractor
            extractor = SchemaFromTextExtractor(
                driver=neo4j_driver,
                llm=OpenAILLM(
                    model_name="gpt-4o",
                    model_params={
                        "max_tokens": 2000,
                        "response_format": {"type": "json_object"},
                    },
                )
            )

            # Extract schema from text - automatically enhanced for database compatibility
            text = "John works at Acme Corp. His email is john@acme.com"
            schema = await extractor.run(text)
            
            # The schema will include entities and properties extracted by the LLM,
            # plus any additional properties/types required by database constraints
    """

    def __init__(
        self,
        driver: neo4j.Driver,
        llm: LLMInterface,
        prompt_template: Optional[PromptTemplate] = None,
        llm_params: Optional[Dict[str, Any]] = None,
        neo4j_database: Optional[str] = None,
    ) -> None:
        """
        Initialize the SchemaFromTextExtractor.
        
        Args:
            driver: Neo4j driver instance for database access
            llm: LLM instance implementing LLMInterface for schema extraction
            prompt_template: Optional custom prompt template for schema extraction.
                            Defaults to SchemaExtractionTemplate.
            llm_params: Optional dictionary of additional parameters to pass to the LLM
            neo4j_database: Optional Neo4j database name. If None, uses default database.
        """
        super().__init__(driver, neo4j_database)
        self._llm: LLMInterface = llm
        self._prompt_template: PromptTemplate = (
            prompt_template or SchemaExtractionTemplate()
        )
        self._llm_params: dict[str, Any] = llm_params or {}

    @validate_call
    async def run(self, text: str, examples: str = "", **kwargs: Any) -> GraphSchema:
        """
        Asynchronously extracts schema from text and enhances it for database compatibility.
        
        This method uses the configured LLM to analyze the provided text and extract
        entity types, relationship types, and their properties. The extracted schema
        is then automatically enhanced to match database constraints.

        Args:
            text: The text from which the schema will be inferred
            examples: Optional examples to guide schema extraction (few-shot learning)
            **kwargs: Additional parameters for schema configuration including:
                - additional_node_types (bool): Whether to allow additional node types (default: True)
                - additional_relationship_types (bool): Whether to allow additional relationship types (default: True)
                
        Returns:
            GraphSchema: A configured schema object extracted from text and enhanced
            to match database constraints. The schema includes:
            - Entity types and properties identified by the LLM
            - Additional properties/types required by database constraints
            - Proper type annotations and required flags based on constraints
            
        Raises:
            LLMGenerationError: If the LLM fails to generate a response
            SchemaExtractionError: If the LLM response cannot be parsed or is invalid
            
        Note:
            This component always operates in enhancement mode and will automatically
            resolve any conflicts between the LLM-generated schema and database constraints.
            If enhancement fails, the original LLM-generated schema is returned with a warning.
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

        # Create initial schema from LLM extraction
        initial_schema = GraphSchema.model_validate(
            {
                "node_types": extracted_node_types,
                "relationship_types": extracted_relationship_types,
                "patterns": extracted_patterns,
            }
        )
        
        # Enhance the schema to match database constraints using shared logic
        enhanced_schema = self._process_constraints_against_schema(initial_schema, mode="enhance", **kwargs)
        
        return enhanced_schema
