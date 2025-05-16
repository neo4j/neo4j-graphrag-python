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
import yaml
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

    model_config = ConfigDict(
        frozen=True,
    )


class SchemaEntity(BaseModel):
    """
    Represents a possible node in the graph.
    """

    label: str
    description: str = ""
    properties: list[SchemaProperty] = []

    @classmethod
    def from_text_or_dict(cls, input: EntityInputType) -> Self:
        if isinstance(input, SchemaEntity):
            return input
        if isinstance(input, str):
            return cls(label=input)
        return cls.model_validate(input)


class SchemaRelation(BaseModel):
    """
    Represents a possible relationship between nodes in the graph.
    """

    label: str
    description: str = ""
    properties: list[SchemaProperty] = []

    @classmethod
    def from_text_or_dict(cls, input: RelationInputType) -> Self:
        if isinstance(input, SchemaRelation):
            return input
        if isinstance(input, str):
            return cls(label=input)
        return cls.model_validate(input)


class GraphSchema(DataModel):
    entities: Tuple[SchemaEntity, ...]
    relations: Optional[Tuple[SchemaRelation, ...]] = None
    potential_schema: Optional[Tuple[Tuple[str, str, str], ...]] = None

    _entity_index: dict[str, SchemaEntity] = PrivateAttr()
    _relation_index: dict[str, SchemaRelation] = PrivateAttr()

    model_config = ConfigDict(
        frozen=True,
    )

    @model_validator(mode="after")
    def check_schema(self) -> Self:
        self._entity_index = {e.label: e for e in self.entities}
        self._relation_index = (
            {r.label: r for r in self.relations} if self.relations else {}
        )

        relations = self.relations or tuple()
        potential_schema = self.potential_schema or tuple()

        if potential_schema:
            if not relations:
                raise SchemaValidationError(
                    "Relations must also be provided when using a potential schema."
                )
            for entity1, relation, entity2 in potential_schema:
                if entity1 not in self._entity_index:
                    raise SchemaValidationError(
                        f"Entity '{entity1}' is not defined in the provided entities."
                    )
                if relation not in self._relation_index:
                    raise SchemaValidationError(
                        f"Relation '{relation}' is not defined in the provided relations."
                    )
                if entity2 not in self._entity_index:
                    raise SchemaValidationError(
                        f"Entity '{entity2}' is not defined in the provided entities."
                    )

        return self

    def entity_from_label(self, label: str) -> Optional[SchemaEntity]:
        return self._entity_index.get(label)

    def relation_from_label(self, label: str) -> Optional[SchemaRelation]:
        return self._relation_index.get(label)

    def store_as_json(self, file_path: str) -> None:
        """
        Save the schema configuration to a JSON file.

        Args:
            file_path (str): The path where the schema configuration will be saved.
        """
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    def store_as_yaml(self, file_path: str) -> None:
        """
        Save the schema configuration to a YAML file.

        Args:
            file_path (str): The path where the schema configuration will be saved.
        """
        # create a copy of the data and convert tuples to lists for YAML compatibility
        data = self.model_dump()
        if data.get("entities"):
            data["entities"] = list(data["entities"])
        if data.get("relations"):
            data["relations"] = list(data["relations"])
        if data.get("potential_schema"):
            data["potential_schema"] = [list(item) for item in data["potential_schema"]]

        with open(file_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

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

        if not file_path.exists():
            raise FileNotFoundError(f"Schema file not found: {file_path}")

        if file_path.suffix.lower() in [".json"]:
            return cls.from_json(file_path)
        elif file_path.suffix.lower() in [".yaml", ".yml"]:
            return cls.from_yaml(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. Use .json, .yaml, or .yml"
            )

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> Self:
        """
        Load a schema configuration from a JSON file.

        Args:
            file_path (Union[str, Path]): The path to the JSON schema configuration file.

        Returns:
            GraphSchema: The loaded schema configuration.
        """
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                return cls.model_validate(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {e}")
            except ValidationError as e:
                raise SchemaValidationError(f"Schema validation failed: {e}")

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> Self:
        """
        Load a schema configuration from a YAML file.

        Args:
            file_path (Union[str, Path]): The path to the YAML schema configuration file.

        Returns:
            GraphSchema: The loaded schema configuration.
        """
        with open(file_path, "r") as f:
            try:
                data = yaml.safe_load(f)
                return cls.model_validate(data)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML file: {e}")
            except ValidationError as e:
                raise SchemaValidationError(f"Schema validation failed: {e}")


class SchemaBuilder(Component):
    """
    A builder class for constructing GraphSchema objects from given entities,
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
        entities: Sequence[SchemaEntity],
        relations: Optional[Sequence[SchemaRelation]] = None,
        potential_schema: Optional[Sequence[Tuple[str, str, str]]] = None,
    ) -> GraphSchema:
        """
        Creates a GraphSchema object from Lists of Entity and Relation objects
        and a Dictionary defining potential relationships.

        Args:
            entities (Sequence[SchemaEntity]): List or tuple of SchemaEntity objects.
            relations (Optional[Sequence[SchemaRelation]]): List or tuple of SchemaRelation objects.
            potential_schema (Optional[Sequence[Tuple[str, str, str]]]): List or tuples of triplets: (source_entity_label, relation_label, target_entity_label).

        Returns:
            GraphSchema: A configured schema object.
        """
        try:
            return GraphSchema.model_validate(
                dict(
                    entities=entities,
                    relations=relations,
                    potential_schema=potential_schema,
                )
            )
        except (ValidationError, SchemaValidationError) as e:
            raise SchemaValidationError(e)

    @validate_call
    async def run(
        self,
        entities: Sequence[SchemaEntity],
        relations: Optional[Sequence[SchemaRelation]] = None,
        potential_schema: Optional[Sequence[Tuple[str, str, str]]] = None,
    ) -> GraphSchema:
        """
        Asynchronously constructs and returns a GraphSchema object.

        Args:
            entities (Sequence[SchemaEntity]): List or tuple of SchemaEntity objects.
            relations (Sequence[SchemaRelation]): List or tuple of SchemaRelation objects.
            potential_schema (Optional[Sequence[Tuple[str, str, str]]]): List or tuples of triplets: (source_entity_label, relation_label, target_entity_label).

        Returns:
            GraphSchema: A configured schema object, constructed asynchronously.
        """
        return self.create_schema_model(entities, relations, potential_schema)


class SchemaFromTextExtractor(Component):
    """
    A component for constructing SchemaConfig objects from the output of an LLM after
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

        extracted_entities: List[Dict[str, Any]] = (
            extracted_schema.get("entities") or []
        )
        extracted_relations: Optional[List[Dict[str, Any]]] = extracted_schema.get(
            "relations"
        )
        potential_schema: Optional[List[Tuple[str, str, str]]] = extracted_schema.get(
            "potential_schema"
        )

        return GraphSchema.model_validate(
            {
                "entities": extracted_entities,
                "relations": extracted_relations,
                "potential_schema": potential_schema,
            }
        )
