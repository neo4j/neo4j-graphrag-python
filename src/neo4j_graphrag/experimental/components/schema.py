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
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Sequence
from pathlib import Path

from pydantic import (
    BaseModel,
    PrivateAttr,
    ValidationError,
    model_validator,
    validate_call,
    ConfigDict,
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
from neo4j_graphrag.utils.file_handler import FileHandler


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

    model_config = ConfigDict(
        frozen=True,
    )


class NodeType(BaseModel):
    """
    Represents a possible node in the graph.
    """

    label: str
    description: str = ""
    properties: list[PropertyType] = []

    @model_validator(mode="before")
    @classmethod
    def validate_input_if_string(cls, data: EntityInputType) -> EntityInputType:
        if isinstance(data, str):
            return {"label": data}
        return data


class RelationshipType(BaseModel):
    """
    Represents a possible relationship between nodes in the graph.
    """

    label: str
    description: str = ""
    properties: list[PropertyType] = []

    @model_validator(mode="before")
    @classmethod
    def validate_input_if_string(cls, data: RelationInputType) -> RelationInputType:
        if isinstance(data, str):
            return {"label": data}
        return data


class GraphSchema(DataModel):
    node_types: Tuple[NodeType, ...]
    relationship_types: Optional[Tuple[RelationshipType, ...]] = None
    patterns: Optional[Tuple[Tuple[str, str, str], ...]] = None

    _node_type_index: dict[str, NodeType] = PrivateAttr()
    _relationship_type_index: dict[str, RelationshipType] = PrivateAttr()

    model_config = ConfigDict(
        frozen=True,
    )

    @model_validator(mode="after")
    def check_schema(self) -> Self:
        self._node_type_index = {node.label: node for node in self.node_types}
        self._relationship_type_index = (
            {r.label: r for r in self.relationship_types}
            if self.relationship_types
            else {}
        )

        relationship_types = self.relationship_types or tuple()
        patterns = self.patterns or tuple()

        if patterns:
            if not relationship_types:
                raise SchemaValidationError(
                    "Relations must also be provided when using a potential schema."
                )
            for entity1, relation, entity2 in patterns:
                if entity1 not in self._node_type_index:
                    raise SchemaValidationError(
                        f"Entity '{entity1}' is not defined in the provided entities."
                    )
                if relation not in self._relationship_type_index:
                    raise SchemaValidationError(
                        f"Relation '{relation}' is not defined in the provided relations."
                    )
                if entity2 not in self._node_type_index:
                    raise SchemaValidationError(
                        f"Entity '{entity2}' is not defined in the provided entities."
                    )

        return self

    def node_type_from_label(self, label: str) -> Optional[NodeType]:
        return self._node_type_index.get(label)

    def relationship_type_from_label(self, label: str) -> Optional[RelationshipType]:
        return self._relationship_type_index.get(label)

    def store_as_json(self, file_path: str) -> None:
        """
        Save the schema configuration to a JSON file.

        Args:
            file_path (str): The path where the schema configuration will be saved.
        """
        data = self.model_dump(mode="json")
        file_handler = FileHandler()
        file_handler.write_json(data, file_path)

    def store_as_yaml(self, file_path: str) -> None:
        """
        Save the schema configuration to a YAML file.

        Args:
            file_path (str): The path where the schema configuration will be saved.
        """
        data = self.model_dump(mode="json")
        file_handler = FileHandler()
        file_handler.write_yaml(data, file_path)

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> Self:
        """
        Load a schema configuration from a file (either JSON or YAML).

        The file format is automatically detected based on the file extension.

        Args:
            file_path (Union[str, Path]): The path to the schema configuration file.

        Returns:
            GraphSchema: The loaded schema configuration.
        """
        file_path = Path(file_path)
        file_handler = FileHandler()
        try:
            data = file_handler.read(file_path)
        except ValueError:
            raise

        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise SchemaValidationError(str(e)) from e


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

    @staticmethod
    def create_schema_model(
        node_types: Sequence[NodeType],
        relationship_types: Optional[Sequence[RelationshipType]] = None,
        patterns: Optional[Sequence[Tuple[str, str, str]]] = None,
    ) -> GraphSchema:
        """
        Creates a GraphSchema object from Lists of Entity and Relation objects
        and a Dictionary defining potential relationships.

        Args:
            node_types (Sequence[NodeType]): List or tuple of NodeType objects.
            relationship_types (Optional[Sequence[RelationshipType]]): List or tuple of RelationshipType objects.
            patterns (Optional[Sequence[Tuple[str, str, str]]]): List or tuples of triplets: (source_entity_label, relation_label, target_entity_label).

        Returns:
            GraphSchema: A configured schema object.
        """
        try:
            return GraphSchema.model_validate(
                dict(
                    node_types=node_types,
                    relationship_types=relationship_types,
                    patterns=patterns,
                )
            )
        except (ValidationError, SchemaValidationError) as e:
            raise SchemaValidationError(e)

    @validate_call
    async def run(
        self,
        node_types: Sequence[NodeType],
        relationship_types: Optional[Sequence[RelationshipType]] = None,
        patterns: Optional[Sequence[Tuple[str, str, str]]] = None,
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
        return self.create_schema_model(node_types, relationship_types, patterns)


class SchemaFromTextExtractor(Component):
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
