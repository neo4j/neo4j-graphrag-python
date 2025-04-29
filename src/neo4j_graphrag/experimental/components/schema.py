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
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pathlib import Path

from pydantic import BaseModel, ValidationError, model_validator, validate_call
from typing_extensions import Self

from neo4j_graphrag.exceptions import SchemaValidationError
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


class SchemaEntity(BaseModel):
    """
    Represents a possible node in the graph.
    """

    label: str
    description: str = ""
    properties: List[SchemaProperty] = []

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
    properties: List[SchemaProperty] = []

    @classmethod
    def from_text_or_dict(cls, input: RelationInputType) -> Self:
        if isinstance(input, SchemaRelation):
            return input
        if isinstance(input, str):
            return cls(label=input)
        return cls.model_validate(input)


class SchemaConfig(DataModel):
    """
    Represents possible relationships between entities and relations in the graph.
    """

    entities: Dict[str, Dict[str, Any]]
    relations: Optional[Dict[str, Dict[str, Any]]]
    potential_schema: Optional[List[Tuple[str, str, str]]]

    @model_validator(mode="before")
    def check_schema(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        entities = data.get("entities", {}).keys()
        relations = data.get("relations", {}).keys()
        potential_schema = data.get("potential_schema", [])

        if potential_schema:
            if not relations:
                raise SchemaValidationError(
                    "Relations must also be provided when using a potential schema."
                )
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
            SchemaConfig: The loaded schema configuration.
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
            SchemaConfig: The loaded schema configuration.
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
            SchemaConfig: The loaded schema configuration.
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
        relations: Optional[List[SchemaRelation]] = None,
        potential_schema: Optional[List[Tuple[str, str, str]]] = None,
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
        relation_dict = (
            {relation.label: relation.model_dump() for relation in relations}
            if relations
            else {}
        )

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
        relations: Optional[List[SchemaRelation]] = None,
        potential_schema: Optional[List[Tuple[str, str, str]]] = None,
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


class SchemaFromText(Component):
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
    async def run(self, text: str, examples: str = "", **kwargs: Any) -> SchemaConfig:
        """
        Asynchronously extracts the schema from text and returns a SchemaConfig object.

        Args:
            text (str): the text from which the schema will be inferred.
            examples (str): examples to guide schema extraction.
        Returns:
            SchemaConfig: A configured schema object, extracted automatically and
            constructed asynchronously.
        """
        prompt: str = self._prompt_template.format(text=text, examples=examples)

        response = await self._llm.ainvoke(prompt, **self._llm_params)
        content: str = (
            response
            if isinstance(response, str)
            else getattr(response, "content", str(response))
        )

        try:
            extracted_schema: Dict[str, Any] = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response is not valid JSON.") from exc

        extracted_entities: List[Dict[str, Any]] = (
            extracted_schema.get("entities") or []
        )
        extracted_relations: Optional[List[Dict[str, Any]]] = extracted_schema.get(
            "relations"
        )
        potential_schema: Optional[List[Tuple[str, str, str]]] = extracted_schema.get(
            "potential_schema"
        )

        entities: List[SchemaEntity] = [SchemaEntity(**e) for e in extracted_entities]
        relations: Optional[List[SchemaRelation]] = (
            [SchemaRelation(**r) for r in extracted_relations]
            if extracted_relations
            else None
        )

        return SchemaBuilder.create_schema_model(
            entities=entities,
            relations=relations,
            potential_schema=potential_schema,
        )
