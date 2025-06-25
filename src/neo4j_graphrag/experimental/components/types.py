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

import enum
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, Literal, Iterable
from typing_extensions import Self

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ValidationError,
    model_validator,
    ConfigDict,
    PrivateAttr,
)

from neo4j_graphrag.exceptions import SchemaValidationError
from neo4j_graphrag.experimental.pipeline.component import DataModel
from neo4j_graphrag.experimental.pipeline.types.schema import (
    RelationInputType,
    EntityInputType,
)
from neo4j_graphrag.utils.file_handler import FileHandler, FileFormat


class DocumentInfo(DataModel):
    """A document loaded by a DataLoader.

    Attributes:
        path (str): Document path.
        metadata (Optional[dict[str, Any]]): Metadata associated with this document.
        uid (str): Unique identifier for this document.
    """

    path: str
    metadata: Optional[Dict[str, str]] = None
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def document_id(self) -> str:
        return self.uid


class PdfDocument(DataModel):
    text: str
    document_info: DocumentInfo


class TextChunk(BaseModel):
    """A chunk of text split from a document by a text splitter.

    Attributes:
        text (str): The raw chunk text.
        index (int): The position of this chunk in the original document.
        metadata (Optional[dict[str, Any]]): Metadata associated with this chunk.
        uid (str): Unique identifier for this chunk.
    """

    text: str
    index: int
    metadata: Optional[dict[str, Any]] = None
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def chunk_id(self) -> str:
        return self.uid


class TextChunks(DataModel):
    """A collection of text chunks returned from a text splitter.

    Attributes:
        chunks (list[TextChunk]): A list of text chunks.
    """

    chunks: list[TextChunk]


class Neo4jNode(BaseModel):
    """Represents a Neo4j node.

    Attributes:
        id (str): The element ID of the node.
        label (str): The label of the node.
        properties (dict[str, Any]): A dictionary of properties attached to the node.
        embedding_properties (Optional[dict[str, list[float]]]): A list of embedding properties attached to the node.
    """

    id: str
    label: str
    properties: dict[str, Any] = {}
    embedding_properties: Optional[dict[str, list[float]]] = None

    @field_validator("properties", "embedding_properties")
    @classmethod
    def check_for_id_properties(
        cls, v: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        if v and "id" in v.keys():
            raise TypeError("'id' as a property name is not allowed")
        return v

    @property
    def token(self) -> str:
        return self.label


class Neo4jRelationship(BaseModel):
    """Represents a Neo4j relationship.

    Attributes:
        start_node_id (str): The ID of the start node.
        end_node_id (str): The ID of the end node.
        type (str): The relationship type.
        properties (dict[str, Any]): A dictionary of properties attached to the relationship.
        embedding_properties (Optional[dict[str, list[float]]]): A list of embedding properties attached to the relationship.
    """

    start_node_id: str
    end_node_id: str
    type: str
    properties: dict[str, Any] = {}
    embedding_properties: Optional[dict[str, list[float]]] = None

    @property
    def token(self) -> str:
        return self.type


class Neo4jGraph(DataModel):
    """Represents a Neo4j graph.

    Attributes:
        nodes (list[Neo4jNode]): A list of nodes in the graph.
        relationships (list[Neo4jRelationship]): A list of relationships in the graph.
    """

    nodes: list[Neo4jNode] = []
    relationships: list[Neo4jRelationship] = []


class ResolutionStats(DataModel):
    number_of_nodes_to_resolve: int
    number_of_created_nodes: Optional[int] = None


DEFAULT_DOCUMENT_NODE_LABEL = "Document"
DEFAULT_CHUNK_NODE_LABEL = "Chunk"
DEFAULT_CHUNK_TO_DOCUMENT_RELATIONSHIP_TYPE = "FROM_DOCUMENT"
DEFAULT_NEXT_CHUNK_RELATIONSHIP_TYPE = "NEXT_CHUNK"
DEFAULT_NODE_TO_CHUNK_RELATIONSHIP_TYPE = "FROM_CHUNK"
DEFAULT_CHUNK_ID_PROPERTY = "id"
DEFAULT_CHUNK_INDEX_PROPERTY = "index"
DEFAULT_CHUNK_TEXT_PROPERTY = "text"
DEFAULT_CHUNK_EMBEDDING_PROPERTY = "embedding"


class LexicalGraphConfig(BaseModel):
    """Configure all labels and property names in the lexical graph."""

    id_prefix: str = Field(deprecated=True, default="")
    document_node_label: str = DEFAULT_DOCUMENT_NODE_LABEL
    chunk_node_label: str = DEFAULT_CHUNK_NODE_LABEL
    chunk_to_document_relationship_type: str = (
        DEFAULT_CHUNK_TO_DOCUMENT_RELATIONSHIP_TYPE
    )
    next_chunk_relationship_type: str = DEFAULT_NEXT_CHUNK_RELATIONSHIP_TYPE
    node_to_chunk_relationship_type: str = DEFAULT_NODE_TO_CHUNK_RELATIONSHIP_TYPE
    chunk_id_property: str = DEFAULT_CHUNK_ID_PROPERTY
    chunk_index_property: str = DEFAULT_CHUNK_INDEX_PROPERTY
    chunk_text_property: str = DEFAULT_CHUNK_TEXT_PROPERTY
    chunk_embedding_property: str = DEFAULT_CHUNK_EMBEDDING_PROPERTY

    @property
    def lexical_graph_node_labels(self) -> tuple[str, ...]:
        return self.document_node_label, self.chunk_node_label

    @property
    def lexical_graph_relationship_types(self) -> tuple[str, ...]:
        return (
            self.chunk_to_document_relationship_type,
            self.next_chunk_relationship_type,
            self.node_to_chunk_relationship_type,
        )


class GraphResult(DataModel):
    graph: Neo4jGraph
    config: LexicalGraphConfig


class Neo4jPropertyType(str, enum.Enum):
    # See https://neo4j.com/docs/cypher-manual/current/values-and-types/property-structural-constructed/#property-types
    BOOLEAN = "BOOLEAN"
    DATE = "DATE"
    DURATION = "DURATION"
    FLOAT = "FLOAT"
    INTEGER = "INTEGER"
    LIST = "LIST"
    LOCAL_DATETIME = "LOCAL_DATETIME"
    LOCAL_TIME = "LOCAL_TIME"
    POINT = "POINT"
    STRING = "STRING"
    ZONED_DATETIME = "ZONED_DATETIME"
    ZONED_DATE = "ZONED_DATE"


class PropertyType(BaseModel):
    """
    Represents a property on a node or relationship in the graph.
    """

    name: str
    type: Neo4jPropertyType | list[Neo4jPropertyType]
    description: str = ""
    required: bool = False


class Neo4jConstraintTypeEnum(str, enum.Enum):
    # see: https://neo4j.com/docs/cypher-manual/current/constraints/
    NODE_KEY = "NODE_KEY"
    UNIQUENESS = "UNIQUENESS"
    NODE_PROPERTY_EXISTENCE = "NODE_PROPERTY_EXISTENCE"
    NODE_PROPERTY_UNIQUENESS = "NODE_PROPERTY_UNIQUENESS"
    NODE_PROPERTY_TYPE = "NODE_PROPERTY_TYPE"
    RELATIONSHIP_KEY = "RELATIONSHIP_KEY"
    RELATIONSHIP_UNIQUENESS = "RELATIONSHIP_UNIQUENESS"
    RELATIONSHIP_PROPERTY_EXISTENCE = "RELATIONSHIP_PROPERTY_EXISTENCE"
    RELATIONSHIP_PROPERTY_UNIQUENESS = "RELATIONSHIP_PROPERTY_UNIQUENESS"
    RELATIONSHIP_PROPERTY_TYPE = "RELATIONSHIP_PROPERTY_TYPE"


class SchemaConstraint(BaseModel):
    """Constraints that can be applied either on a node or relationship property."""

    entity_type: Literal["NODE", "RELATIONSHIP"]
    label_or_type: list[str]
    type: Neo4jConstraintTypeEnum
    properties: list[str]
    property_type: Optional[list[Neo4jPropertyType]] = None
    name: Optional[str] = None  # do not force users to set a name manually

    @field_validator("label_or_type", mode="before")
    @classmethod
    def _validate_label_or_type(cls, v: Any) -> Iterable[Any]:
        if isinstance(v, str) or not isinstance(v, Iterable):
            return [v]
        return v

    @field_validator("properties", mode="before")
    @classmethod
    def _validate_properties(cls, v: Any) -> Iterable[Any]:
        if isinstance(v, str) or not isinstance(v, Iterable):
            return [v]
        return v


class GraphEntityType(BaseModel):
    """Represents a possible entity in the graph (node or relationship).

    They have a label and a list of properties.

    For LLM-based applications, it is also useful to add a description.

    The additional_properties flag is used in schema-driven data validation.
    """

    label: str
    description: str = ""
    properties: list[PropertyType] = []
    additional_properties: bool = True

    _entity_type_name: Literal["NODE", "RELATIONSHIP"] = PrivateAttr()

    @model_validator(mode="after")
    def validate_additional_properties(self) -> Self:
        if len(self.properties) == 0 and not self.additional_properties:
            raise ValueError(
                "Using `additional_properties=False` with no defined "
                "properties will cause the model to be pruned during graph cleaning.",
            )
        return self

    def get_property_by_name(self, name: str) -> PropertyType | None:
        for prop in self.properties:
            if prop.name == name:
                return prop
        return None

    @property
    def entity_type_name(self) -> Literal["NODE", "RELATIONSHIP"]:
        """Get the entity type name."""
        return self._entity_type_name

    @staticmethod
    def unique_constraint_name() -> tuple[Neo4jConstraintTypeEnum, ...]:
        raise NotImplementedError()


class NodeType(GraphEntityType):
    """Represents a possible node in the graph."""

    _entity_type_name: Literal["NODE", "RELATIONSHIP"] = PrivateAttr(default="NODE")

    @model_validator(mode="before")
    @classmethod
    def validate_input_if_string(cls, data: EntityInputType) -> EntityInputType:
        if isinstance(data, str):
            return {"label": data}
        return data

    @staticmethod
    def unique_constraint_name() -> tuple[Neo4jConstraintTypeEnum, ...]:
        return (
            Neo4jConstraintTypeEnum.NODE_KEY,
            Neo4jConstraintTypeEnum.UNIQUENESS,
        )


class RelationshipType(GraphEntityType):
    """Represents a possible relationship between two nodes in the graph."""

    _entity_type_name: Literal["NODE", "RELATIONSHIP"] = PrivateAttr(default="RELATIONSHIP")

    @model_validator(mode="before")
    @classmethod
    def validate_input_if_string(cls, data: RelationInputType) -> RelationInputType:
        if isinstance(data, str):
            return {"label": data}
        return data

    @staticmethod
    def unique_constraint_name() -> tuple[Neo4jConstraintTypeEnum, ...]:
        return (
            Neo4jConstraintTypeEnum.RELATIONSHIP_KEY,
            Neo4jConstraintTypeEnum.RELATIONSHIP_UNIQUENESS,
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
    constraints: Tuple[SchemaConstraint, ...] = tuple()

    additional_node_types: bool = True
    additional_relationship_types: bool = True
    additional_patterns: bool = True

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
    def validate_constraint_on_properties(self) -> Self:
        """Check that properties in constraints are listed in the property list."""
        for c in self.constraints:
            entity: GraphEntityType | None = None
            if c.entity_type == "NODE":
                entity = self.node_type_from_label(c.label_or_type)
            else:
                entity = self.relationship_type_from_label(c.label_or_type)
            if not entity:
                raise ValueError(f"Entity type {c.label_or_type} is not defined.")
            allowed_prop_names = [p.name for p in entity.properties]
            for prop_name in c.properties:
                if prop_name not in allowed_prop_names:
                    raise ValueError(
                        f"Property '{prop_name}' has a constraint '{c}' but is not in the property list for entity {entity}."
                    )
        return self

    def node_type_from_label(self, label: str) -> Optional[NodeType]:
        return self._node_type_index.get(label)

    def relationship_type_from_label(self, label: str) -> Optional[RelationshipType]:
        return self._relationship_type_index.get(label)

    def unique_properties_for_entity(self, entity: GraphEntityType) -> list[list[str]]:
        result = []
        for c in self.constraints:
            if c.entity_type != entity.entity_type_name:
                continue
            if c.label_or_type != entity.label:
                continue
            if c.type in entity.unique_constraint_name():
                result.append(c.properties)
        return result

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
