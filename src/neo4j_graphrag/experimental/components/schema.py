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

import json
import re

import neo4j
import logging
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Sequence, Callable
from pathlib import Path

from pydantic import (
    BaseModel,
    PrivateAttr,
    model_validator,
    validate_call,
    ConfigDict,
    ValidationError,
    Field,
)
from typing_extensions import Self

from neo4j_graphrag.exceptions import (
    SchemaValidationError,
    LLMGenerationError,
    SchemaExtractionError,
)
from neo4j_graphrag.experimental.pipeline.component import Component, DataModel
from neo4j_graphrag.experimental.pipeline.types.schema import (
    EntityInputType,
    RelationInputType,
)
from neo4j_graphrag.generation import SchemaExtractionTemplate, PromptTemplate
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.utils.file_handler import FileHandler, FileFormat
from neo4j_graphrag.schema import get_structured_schema


logger = logging.getLogger(__name__)


class PropertyType(BaseModel):
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
    required: bool = False

    model_config = ConfigDict(
        frozen=True,
    )


def default_additional_item(key: str) -> Callable[[dict[str, Any]], bool]:
    def wrapper(validated_data: dict[str, Any]) -> bool:
        return len(validated_data.get(key, [])) == 0

    return wrapper


class NodeType(BaseModel):
    """
    Represents a possible node in the graph.
    """

    label: str
    description: str = ""
    properties: list[PropertyType] = []
    additional_properties: bool = Field(
        default_factory=default_additional_item("properties")
    )

    @model_validator(mode="before")
    @classmethod
    def validate_input_if_string(cls, data: EntityInputType) -> EntityInputType:
        if isinstance(data, str):
            return {"label": data}
        return data

    @model_validator(mode="after")
    def validate_additional_properties(self) -> Self:
        if len(self.properties) == 0 and not self.additional_properties:
            raise ValueError(
                "Using `additional_properties=False` with no defined "
                "properties will cause the model to be pruned during graph cleaning. "
                f"Define some properties or remove this NodeType: {self}"
            )
        return self

    def property_type_from_name(self, name: str) -> Optional[PropertyType]:
        for prop in self.properties:
            if prop.name == name:
                return prop
        return None


class RelationshipType(BaseModel):
    """
    Represents a possible relationship between nodes in the graph.
    """

    label: str
    description: str = ""
    properties: list[PropertyType] = []
    additional_properties: bool = Field(
        default_factory=default_additional_item("properties")
    )

    @model_validator(mode="before")
    @classmethod
    def validate_input_if_string(cls, data: RelationInputType) -> RelationInputType:
        if isinstance(data, str):
            return {"label": data}
        return data

    @model_validator(mode="after")
    def validate_additional_properties(self) -> Self:
        if len(self.properties) == 0 and not self.additional_properties:
            raise ValueError(
                "Using `additional_properties=False` with no defined "
                "properties will cause the model to be pruned during graph cleaning. "
                f"Define some properties or remove this RelationshipType: {self}"
            )
        return self

    def property_type_from_name(self, name: str) -> Optional[PropertyType]:
        for prop in self.properties:
            if prop.name == name:
                return prop
        return None


class ConstraintType(BaseModel):
    """
    Represents a constraint on a node in the graph.
    """

    type: Literal[
        "UNIQUENESS"
    ]  # TODO: add other constraint types ["propertyExistence", "propertyType", "key"]
    node_type: str
    property_name: str

    model_config = ConfigDict(
        frozen=True,
    )


class GraphSchema(DataModel):
    """This model represents the expected
    node and relationship types in the graph.

    It is used both for guiding the LLM in the entity and relation
    extraction component, and for cleaning the extracted graph in a
    post-processing step.

    .. warning::

        This model is immutable.
    """

    node_types: Tuple[NodeType, ...]
    relationship_types: Tuple[RelationshipType, ...] = tuple()
    patterns: Tuple[Tuple[str, str, str], ...] = tuple()
    constraints: Tuple[ConstraintType, ...] = tuple()

    additional_node_types: bool = Field(
        default_factory=default_additional_item("node_types")
    )
    additional_relationship_types: bool = Field(
        default_factory=default_additional_item("relationship_types")
    )
    additional_patterns: bool = Field(
        default_factory=default_additional_item("patterns")
    )

    _node_type_index: dict[str, NodeType] = PrivateAttr()
    _relationship_type_index: dict[str, RelationshipType] = PrivateAttr()

    model_config = ConfigDict(
        frozen=True,
    )

    @model_validator(mode="after")
    def validate_patterns_against_node_and_rel_types(self) -> Self:
        self._node_type_index = {node.label: node for node in self.node_types}
        self._relationship_type_index = (
            {r.label: r for r in self.relationship_types}
            if self.relationship_types
            else {}
        )

        relationship_types = self.relationship_types
        patterns = self.patterns

        if patterns:
            if not relationship_types:
                raise SchemaValidationError(
                    "Relationship types must also be provided when using patterns."
                )
            for entity1, relation, entity2 in patterns:
                if entity1 not in self._node_type_index:
                    raise SchemaValidationError(
                        f"Node type '{entity1}' is not defined in the provided node_types."
                    )
                if relation not in self._relationship_type_index:
                    raise SchemaValidationError(
                        f"Relationship type '{relation}' is not defined in the provided relationship_types."
                    )
                if entity2 not in self._node_type_index:
                    raise ValueError(
                        f"Node type '{entity2}' is not defined in the provided node_types."
                    )

        return self

    @model_validator(mode="after")
    def validate_additional_parameters(self) -> Self:
        if (
            self.additional_patterns is False
            and self.additional_relationship_types is True
        ):
            raise ValueError(
                "`additional_relationship_types` must be set to False when using `additional_patterns=False`"
            )
        return self

    @model_validator(mode="after")
    def validate_constraints_against_node_types(self) -> Self:
        if not self.constraints:
            return self
        for constraint in self.constraints:
            if not constraint.property_name:
                raise SchemaValidationError(
                    f"Constraint has no property name: {constraint}. Property name is required."
                )
            if constraint.node_type not in self._node_type_index:
                raise SchemaValidationError(
                    f"Constraint references undefined node type: {constraint.node_type}"
                )
        return self

    def node_type_from_label(self, label: str) -> Optional[NodeType]:
        return self._node_type_index.get(label)

    def relationship_type_from_label(self, label: str) -> Optional[RelationshipType]:
        return self._relationship_type_index.get(label)

    @classmethod
    def create_empty(cls) -> Self:
        return cls(node_types=tuple())

    def save(
        self,
        file_path: Union[str, Path],
        overwrite: bool = False,
        format: Optional[FileFormat] = None,
    ) -> None:
        """
        Save the schema configuration to file.

        Args:
            file_path (str): The path where the schema configuration will be saved.
            overwrite (bool): If set to True, existing file will be overwritten. Default to False.
            format (Optional[FileFormat]): The file format to save the schema configuration into. By default, it is inferred from file_path extension.
        """
        data = self.model_dump(mode="json")
        file_handler = FileHandler()
        file_handler.write(data, file_path, overwrite=overwrite, format=format)

    def store_as_json(
        self, file_path: Union[str, Path], overwrite: bool = False
    ) -> None:
        warnings.warn(
            "Use .save(..., format=FileFormat.JSON) instead.", DeprecationWarning
        )
        return self.save(file_path, overwrite=overwrite, format=FileFormat.JSON)

    def store_as_yaml(
        self, file_path: Union[str, Path], overwrite: bool = False
    ) -> None:
        warnings.warn(
            "Use .save(..., format=FileFormat.YAML) instead.", DeprecationWarning
        )
        return self.save(file_path, overwrite=overwrite, format=FileFormat.YAML)

    @classmethod
    def from_file(
        cls, file_path: Union[str, Path], format: Optional[FileFormat] = None
    ) -> Self:
        """
        Load a schema configuration from a file (either JSON or YAML).

        The file format is automatically detected based on the file extension,
        unless the format parameter is set.

        Args:
            file_path (Union[str, Path]): The path to the schema configuration file.
            format (Optional[FileFormat]): The format of the schema configuration file (json or yaml).

        Returns:
            GraphSchema: The loaded schema configuration.
        """
        file_path = Path(file_path)
        file_handler = FileHandler()
        try:
            data = file_handler.read(file_path, format=format)
        except ValueError:
            raise

        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise SchemaValidationError(str(e)) from e


class BaseSchemaBuilder(Component):
    async def run(self, *args: Any, **kwargs: Any) -> GraphSchema:
        raise NotImplementedError()


class SchemaBuilder(BaseSchemaBuilder):
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

    @staticmethod
    def create_schema_model(
        node_types: Sequence[NodeType],
        relationship_types: Optional[Sequence[RelationshipType]] = None,
        patterns: Optional[Sequence[Tuple[str, str, str]]] = None,
        constraints: Optional[Sequence[ConstraintType]] = None,
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
            return GraphSchema.model_validate(
                dict(
                    node_types=node_types,
                    relationship_types=relationship_types or (),
                    patterns=patterns or (),
                    constraints=constraints or (),
                    **kwargs,
                )
            )
        except ValidationError as e:
            raise SchemaValidationError() from e

    @validate_call
    async def run(
        self,
        node_types: Sequence[NodeType],
        relationship_types: Optional[Sequence[RelationshipType]] = None,
        patterns: Optional[Sequence[Tuple[str, str, str]]] = None,
        constraints: Optional[Sequence[ConstraintType]] = None,
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
        return self.create_schema_model(
            node_types,
            relationship_types,
            patterns,
            constraints,
            **kwargs,
        )


class SchemaFromTextExtractor(BaseSchemaBuilder):
    """
    A component for constructing GraphSchema objects from the output of an LLM after
    automatic schema extraction from text.
    """

    def __init__(
        self,
        llm: LLMInterface,
        prompt_template: Optional[PromptTemplate] = None,
        llm_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._llm: LLMInterface = llm
        self._prompt_template: PromptTemplate = (
            prompt_template or SchemaExtractionTemplate()
        )
        self._llm_params: dict[str, Any] = llm_params or {}

    def _filter_invalid_patterns(
        self,
        patterns: List[Tuple[str, str, str]],
        node_types: List[Dict[str, Any]],
        relationship_types: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[str, str, str]]:
        """
        Filter out patterns that reference undefined node types or relationship types.

        Args:
            patterns: List of patterns to filter.
            node_types: List of node type definitions.
            relationship_types: Optional list of relationship type definitions.

        Returns:
            Filtered list of patterns containing only valid references.
        """
        # Early returns for missing required types
        if not node_types:
            logging.info(
                "Filtering out all patterns because no node types are defined. "
                "Patterns reference node types that must be defined."
            )
            return []

        if not relationship_types:
            logging.info(
                "Filtering out all patterns because no relationship types are defined. "
                "GraphSchema validation requires relationship_types when patterns are provided."
            )
            return []

        # Create sets of valid labels
        valid_node_labels = {node_type["label"] for node_type in node_types}
        valid_relationship_labels = {
            rel_type["label"] for rel_type in relationship_types
        }

        # Filter patterns
        filtered_patterns = []
        for pattern in patterns:
            if not (isinstance(pattern, (list, tuple)) and len(pattern) == 3):
                continue

            entity1, relation, entity2 = pattern

            # Check if all components are valid
            if (
                entity1 in valid_node_labels
                and entity2 in valid_node_labels
                and relation in valid_relationship_labels
            ):
                filtered_patterns.append(pattern)
            else:
                # Log invalid pattern with validation details
                entity1_valid = entity1 in valid_node_labels
                entity2_valid = entity2 in valid_node_labels
                relation_valid = relation in valid_relationship_labels

                logging.info(
                    f"Filtering out invalid pattern: {pattern}. "
                    f"Entity1 '{entity1}' valid: {entity1_valid}, "
                    f"Entity2 '{entity2}' valid: {entity2_valid}, "
                    f"Relation '{relation}' valid: {relation_valid}"
                )

        return filtered_patterns

    def _filter_items_without_labels(
        self, items: List[Dict[str, Any]], item_type: str
    ) -> List[Dict[str, Any]]:
        """Filter out items that have no valid labels."""
        filtered_items = []
        for item in items:
            if isinstance(item, str):
                if item and " " not in item and not item.startswith("{"):
                    filtered_items.append({"label": item})
                elif item:
                    logging.info(
                        f"Filtering out {item_type} with invalid label: {item}"
                    )
            elif isinstance(item, dict) and item.get("label"):
                filtered_items.append(item)
            else:
                logging.info(f"Filtering out {item_type} with missing label: {item}")
        return filtered_items

    def _filter_nodes_without_labels(
        self, node_types: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter out node types that have no labels."""
        return self._filter_items_without_labels(node_types, "node type")

    def _filter_relationships_without_labels(
        self, relationship_types: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter out relationship types that have no labels."""
        return self._filter_items_without_labels(
            relationship_types, "relationship type"
        )

    def _filter_invalid_constraints(
        self, constraints: List[Dict[str, Any]], node_types: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter out constraints that reference undefined node types or have no property name."""
        if not constraints:
            return []

        if not node_types:
            logging.info(
                "Filtering out all constraints because no node types are defined. "
                "Constraints reference node types that must be defined."
            )
            return []

        valid_node_labels = {node_type.get("label") for node_type in node_types}

        filtered_constraints = []
        for constraint in constraints:
            # check if the property_name is provided
            if not constraint.get("property_name"):
                logging.info(
                    f"Filtering out constraint: {constraint}. "
                    f"Property name is not provided."
                )
                continue
            # check if the node_type is valid
            if constraint.get("node_type") not in valid_node_labels:
                logging.info(
                    f"Filtering out constraint: {constraint}. "
                    f"Node type '{constraint.get('node_type')}' is not valid. Valid node types: {valid_node_labels}"
                )
                continue
            filtered_constraints.append(constraint)
        return filtered_constraints

    def _clean_json_content(self, content: str) -> str:
        content = content.strip()

        # Remove markdown code block markers if present
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
        content = re.sub(r"```\s*$", "", content, flags=re.MULTILINE)

        return content.strip()

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

        # Clean response
        content = self._clean_json_content(content)

        try:
            extracted_schema: Dict[str, Any] = json.loads(content)

            # handle dictionary
            if isinstance(extracted_schema, dict):
                pass  # Keep as is
            # handle list
            elif isinstance(extracted_schema, list):
                if len(extracted_schema) == 0:
                    logging.info(
                        "LLM returned an empty list for schema. Falling back to empty schema."
                    )
                    extracted_schema = {}
                elif isinstance(extracted_schema[0], dict):
                    extracted_schema = extracted_schema[0]
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
        extracted_constraints: Optional[List[Dict[str, Any]]] = extracted_schema.get(
            "constraints"
        )

        # Filter out nodes and relationships without labels
        extracted_node_types = self._filter_nodes_without_labels(extracted_node_types)
        if extracted_relationship_types:
            extracted_relationship_types = self._filter_relationships_without_labels(
                extracted_relationship_types
            )

        # Filter out invalid patterns before validation
        if extracted_patterns:
            extracted_patterns = self._filter_invalid_patterns(
                extracted_patterns, extracted_node_types, extracted_relationship_types
            )

        # Filter out invalid constraints
        if extracted_constraints:
            extracted_constraints = self._filter_invalid_constraints(
                extracted_constraints, extracted_node_types
            )

        return GraphSchema.model_validate(
            {
                "node_types": extracted_node_types,
                "relationship_types": extracted_relationship_types,
                "patterns": extracted_patterns,
                "constraints": extracted_constraints or [],
            }
        )


class SchemaFromExistingGraphExtractor(BaseSchemaBuilder):
    """A class to build a GraphSchema object from an existing graph.

     Uses the get_structured_schema function to extract existing node labels,
     relationship types, properties and existence constraints.

     By default, the built schema does not allow any additional item (property,
     node label, relationship type or pattern).

    Args:
         driver (neo4j.Driver): connection to the neo4j database.
         additional_properties (bool, default False): see GraphSchema
         additional_node_types (bool, default False): see GraphSchema
         additional_relationship_types (bool, default False): see GraphSchema:
         additional_patterns (bool, default False): see GraphSchema:
         neo4j_database (Optional | str): name of the neo4j database to use
    """

    def __init__(
        self,
        driver: neo4j.Driver,
        additional_properties: bool | None = None,
        additional_node_types: bool | None = None,
        additional_relationship_types: bool | None = None,
        additional_patterns: bool | None = None,
        neo4j_database: Optional[str] = None,
    ) -> None:
        self.driver = driver
        self.database = neo4j_database

        self.additional_properties = additional_properties
        self.additional_node_types = additional_node_types
        self.additional_relationship_types = additional_relationship_types
        self.additional_patterns = additional_patterns

    @staticmethod
    def _extract_required_properties(
        structured_schema: dict[str, Any],
    ) -> list[tuple[str, str]]:
        """Extract a list of (node label (or rel type), property name) for which
         an "EXISTENCE" or "KEY" constraint is defined in the DB.

         Args:

             structured_schema (dict[str, Any]): the result of the `get_structured_schema()` function.

        Returns:

            list of tuples of (node label (or rel type), property name)

        """
        schema_metadata = structured_schema.get("metadata", {})
        existence_constraint = []  # list of (node label, property name)
        for constraint in schema_metadata.get("constraint", []):
            if constraint["type"] in (
                "NODE_PROPERTY_EXISTENCE",
                "NODE_KEY",
                "RELATIONSHIP_PROPERTY_EXISTENCE",
                "RELATIONSHIP_KEY",
            ):
                properties = constraint["properties"]
                labels = constraint["labelsOrTypes"]
                # note: existence constraint only apply to a single property
                # and a single label
                prop = properties[0]
                lab = labels[0]
                existence_constraint.append((lab, prop))
        return existence_constraint

    def _to_schema_entity_dict(
        self,
        key: str,
        property_dict: list[dict[str, Any]],
        existence_constraint: list[tuple[str, str]],
    ) -> dict[str, Any]:
        entity_dict: dict[str, Any] = {
            "label": key,
            "properties": [
                {
                    "name": p["property"],
                    "type": p["type"],
                    "required": (key, p["property"]) in existence_constraint,
                }
                for p in property_dict
            ],
        }
        if self.additional_properties:
            entity_dict["additional_properties"] = self.additional_properties
        return entity_dict

    async def run(self, *args: Any, **kwargs: Any) -> GraphSchema:
        structured_schema = get_structured_schema(self.driver, database=self.database)
        existence_constraint = self._extract_required_properties(structured_schema)

        # node label with properties
        node_labels = set(structured_schema["node_props"].keys())
        node_types = [
            self._to_schema_entity_dict(key, properties, existence_constraint)
            for key, properties in structured_schema["node_props"].items()
        ]

        # relationships with properties
        rel_labels = set(structured_schema["rel_props"].keys())
        relationship_types = [
            self._to_schema_entity_dict(key, properties, existence_constraint)
            for key, properties in structured_schema["rel_props"].items()
        ]

        patterns = [
            (s["start"], s["type"], s["end"])
            for s in structured_schema["relationships"]
        ]

        # deal with nodes and relationships without properties
        for source, rel, target in patterns:
            if source not in node_labels:
                if self.additional_properties is False:
                    logger.warning(
                        f"SCHEMA: found node label {source} without property and additional_properties=False: this node label will always be pruned!"
                    )
                node_labels.add(source)
                node_types.append(
                    {
                        "label": source,
                    }
                )
            if target not in node_labels:
                if self.additional_properties is False:
                    logger.warning(
                        f"SCHEMA: found node label {target} without property and additional_properties=False: this node label will always be pruned!"
                    )
                node_labels.add(target)
                node_types.append(
                    {
                        "label": target,
                    }
                )
            if rel not in rel_labels:
                rel_labels.add(rel)
                relationship_types.append(
                    {
                        "label": rel,
                    }
                )
        schema_dict: dict[str, Any] = {
            "node_types": node_types,
            "relationship_types": relationship_types,
            "patterns": patterns,
        }
        if self.additional_node_types is not None:
            schema_dict["additional_node_types"] = self.additional_node_types
        if self.additional_relationship_types is not None:
            schema_dict["additional_relationship_types"] = (
                self.additional_relationship_types
            )
        if self.additional_patterns is not None:
            schema_dict["additional_patterns"] = self.additional_patterns
        return GraphSchema.model_validate(
            schema_dict,
        )
