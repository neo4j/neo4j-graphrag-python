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
import json
import logging
import re
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
    cast,
)

import neo4j
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
    validate_call,
)
from typing_extensions import Self

from neo4j_graphrag.exceptions import (
    LLMGenerationError,
    SchemaExtractionError,
    SchemaValidationError,
)
from neo4j_graphrag.experimental.pipeline.component import Component, DataModel
from neo4j_graphrag.experimental.pipeline.types.schema import (
    EntityInputType,
    RelationInputType,
)
from neo4j_graphrag.generation import PromptTemplate, SchemaExtractionTemplate
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.schema import get_structured_schema
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.file_handler import FileFormat, FileHandler
from neo4j_graphrag.utils.json_schema_structured_output import (
    make_strict_json_schema_for_structured_output,
)

if TYPE_CHECKING:
    from neo4j_graphrag.experimental.components.graph_schema_extraction import (
        GraphSchemaExtractionOutput,
    )

logger = logging.getLogger(__name__)

# Shared with :class:`ExtractedPropertyType` (schema extraction structured output).
Neo4jPropertyTypeName: TypeAlias = Literal[
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

_DUNDER_RE = re.compile(r"^__|__$")


class GraphConstraintType(str, enum.Enum):
    """Constraint kinds for :class:`ConstraintType`.

    ``UNIQUENESS`` is for node properties only in this API. ``EXISTENCE`` marks a mandatory
    (non-null) node or relationship property, analogous to Neo4j property existence constraints.
    ``KEY`` matches Neo4j NODE KEY / RELATIONSHIP KEY (single property in this API): mandatory
    and unique in the database; this library enforces mandatory presence in pruning only.
    """

    UNIQUENESS = "UNIQUENESS"
    EXISTENCE = "EXISTENCE"
    KEY = "KEY"


def _reject_dunder_label(label: str, kind: str) -> str:
    """Raise ValueError if *label* starts or ends with double underscores."""
    if _DUNDER_RE.search(label):
        raise ValueError(
            f"{kind} label '{label}' uses a reserved '__' prefix or suffix. "
            "This convention is reserved for internal Neo4j GraphRAG labels."
        )
    return label


class PropertyType(BaseModel):
    """
    Represents a property on a node or relationship in the graph.
    """

    name: str
    # See https://neo4j.com/docs/cypher-manual/current/values-and-types/property-structural-constructed/#property-types
    type: Neo4jPropertyTypeName
    description: str = ""
    required: bool = Field(
        default=False,
        deprecated=(
            "Use GraphSchema.constraints with type EXISTENCE instead of PropertyType.required. "
            "Uniqueness does not imply existence; model existence explicitly with EXISTENCE constraints."
        ),
    )
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
    properties: list[PropertyType] = Field(default_factory=list, min_length=1)
    additional_properties: bool = Field(
        default_factory=default_additional_item("properties")
    )

    @field_validator("label")
    @classmethod
    def label_must_not_use_dunder(cls, v: str) -> str:
        return _reject_dunder_label(v, "Node")

    @model_validator(mode="before")
    @classmethod
    def validate_input_if_string(cls, data: EntityInputType) -> EntityInputType:
        if isinstance(data, str):
            logger.info(
                f"Converting string '{data}' to NodeType with default 'name' property "
                f"and additional_properties=True to allow flexible property extraction."
            )
            return {
                "label": data,
                # added to satisfy the model validation (min_length=1 for properties of node types)
                "properties": [{"name": "name", "type": "STRING"}],
                # allow LLM to extract additional properties beyond the default "name"
                "additional_properties": True,  # type: ignore[dict-item]
            }
        if isinstance(data, dict) and "properties" not in data:
            if data.get("additional_properties") is False:  # type: ignore[comparison-overlap]
                return data
            label = data.get("label", "")
            logger.info(
                f"No properties defined for NodeType '{label}'. "
                f"Adding default 'name' property and additional_properties=True "
                f"to allow flexible property extraction."
            )
            return {
                **data,
                # added to satisfy the model validation (min_length=1 for properties of node types)
                "properties": [{"name": "name", "type": "STRING"}],
                # allow LLM to extract additional properties beyond the default "name"
                "additional_properties": True,  # type: ignore[dict-item]
            }

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

    @field_validator("label")
    @classmethod
    def label_must_not_use_dunder(cls, v: str) -> str:
        return _reject_dunder_label(v, "Relationship")

    @model_validator(mode="before")
    @classmethod
    def validate_input_if_string(cls, data: RelationInputType) -> RelationInputType:
        if isinstance(data, str):
            return {"label": data}
        return data

    @model_validator(mode="after")
    def validate_additional_properties(self) -> Self:
        if len(self.properties) == 0 and not self.additional_properties:
            logger.info(
                f"Auto-correcting RelationshipType '{self.label}': "
                f"Setting additional_properties=True because properties list is empty. "
                f"This allows the LLM to extract properties during graph construction."
            )
            self.additional_properties = True
        return self

    def property_type_from_name(self, name: str) -> Optional[PropertyType]:
        for prop in self.properties:
            if prop.name == name:
                return prop
        return None


class ConstraintType(BaseModel):
    """
    Represents a schema-level constraint on a node or relationship property.

    ``UNIQUENESS`` applies to node properties only. ``EXISTENCE`` and ``KEY`` use exactly one of
    ``node_type`` or ``relationship_type`` (non-empty), like Neo4j node vs relationship scope.
    """

    type: GraphConstraintType
    property_name: str
    node_type: str = ""
    relationship_type: Optional[str] = None

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
    )

    @model_validator(mode="after")
    def validate_constraint_shape(self) -> Self:
        if self.type == GraphConstraintType.UNIQUENESS:
            if not (self.node_type and self.node_type.strip()):
                raise ValueError(
                    "UNIQUENESS constraint requires a non-empty node_type; "
                    "relationship uniqueness is not supported on GraphSchema constraints."
                )
            if self.relationship_type is not None and self.relationship_type.strip():
                raise ValueError(
                    "UNIQUENESS constraint must not set relationship_type; "
                    "only node-level UNIQUENESS is supported."
                )
        elif self.type in (
            GraphConstraintType.EXISTENCE,
            GraphConstraintType.KEY,
        ):
            has_node = bool(self.node_type and self.node_type.strip())
            has_rel = bool(self.relationship_type and self.relationship_type.strip())
            if has_node == has_rel:
                raise ValueError(
                    f"{self.type} constraint requires exactly one of node_type or "
                    "relationship_type (non-empty), not both and not neither."
                )
        return self


def _validate_existence_or_key_property_defined(
    constraint: ConstraintType,
    *,
    node_type_index: dict[str, NodeType],
    relationship_type_index: dict[str, RelationshipType],
    kind: Literal["EXISTENCE", "KEY"],
) -> None:
    """Raise SchemaValidationError if the property is not declared on the constrained entity."""
    has_node = bool(constraint.node_type and constraint.node_type.strip())
    has_rel = bool(
        constraint.relationship_type and constraint.relationship_type.strip()
    )
    if has_node:
        if constraint.node_type not in node_type_index:
            raise SchemaValidationError(
                f"Constraint references undefined node type: {constraint.node_type}"
            )
        node_type = node_type_index[constraint.node_type]
        valid_property_names = {p.name for p in node_type.properties}
        if constraint.property_name not in valid_property_names:
            raise SchemaValidationError(
                f"{kind} constraint references undefined property "
                f"'{constraint.property_name}' on node type '{constraint.node_type}'. "
                f"Valid properties: {valid_property_names}"
            )
    elif has_rel:
        rlabel = constraint.relationship_type
        assert rlabel is not None
        if rlabel not in relationship_type_index:
            raise SchemaValidationError(
                f"Constraint references undefined relationship type: {rlabel}"
            )
        rel_type = relationship_type_index[rlabel]
        valid_property_names = {p.name for p in rel_type.properties}
        if constraint.property_name not in valid_property_names:
            raise SchemaValidationError(
                f"{kind} constraint references undefined property "
                f"'{constraint.property_name}' on relationship type '{rlabel}'. "
                f"Valid properties: {valid_property_names}"
            )


class Pattern(BaseModel):
    """Represents a relationship pattern in the graph schema.

    This model provides backward compatibility with tuple-based patterns
    through helper methods (__iter__, __getitem__, __eq__, __hash__).
    """

    source: str
    relationship: str
    target: str

    model_config = ConfigDict(frozen=True)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Allow unpacking: source, rel, target = pattern"""
        return iter((self.source, self.relationship, self.target))

    def __getitem__(self, index: int) -> str:
        """Allow indexing: pattern[0] returns source"""
        return (self.source, self.relationship, self.target)[index]

    def __eq__(self, other: object) -> bool:
        """Allow comparison with tuples for backward compatibility."""
        if isinstance(other, Pattern):
            return (
                self.source,
                self.relationship,
                self.target,
            ) == (
                other.source,
                other.relationship,
                other.target,
            )
        if isinstance(other, (tuple, list)) and len(other) == 3:
            return (self.source, self.relationship, self.target) == tuple(other)
        return False

    def __hash__(self) -> int:
        return hash((self.source, self.relationship, self.target))


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
    patterns: Tuple[Pattern, ...] = tuple()
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

    @model_validator(mode="before")
    @classmethod
    def convert_tuple_patterns(cls, data: Any) -> Any:
        """Convert tuple patterns to Pattern objects for backward compatibility."""
        if isinstance(data, dict) and "patterns" in data and data["patterns"]:
            patterns = data["patterns"]
            converted = []
            for p in patterns:
                if isinstance(p, (tuple, list)) and len(p) == 3:
                    converted.append(
                        Pattern(source=p[0], relationship=p[1], target=p[2])
                    )
                else:
                    converted.append(p)
            data["patterns"] = converted
        return data

    @model_validator(mode="before")
    @classmethod
    def migrate_deprecated_required_to_existence_constraints(cls, data: Any) -> Any:
        """Convert legacy ``PropertyType.required`` to ``EXISTENCE`` constraints and clear flags."""
        if not isinstance(data, dict):
            return data

        def _property_type_required_is_true(prop: PropertyType) -> bool:
            # Use model_dump so we do not access the deprecated ``.required`` attribute.
            return prop.model_dump().get("required") is True

        constraints = list(data.get("constraints") or [])

        def _constraint_identity(c: Any) -> tuple[str, str, str, str]:
            if isinstance(c, ConstraintType):
                nt = c.node_type or ""
                rt = c.relationship_type or ""
                return (str(c.type), nt, rt, c.property_name)
            if isinstance(c, dict):
                return (
                    str(c.get("type", "")),
                    str(c.get("node_type") or ""),
                    str(c.get("relationship_type") or ""),
                    str(c.get("property_name") or ""),
                )
            return ("", "", "", "")

        seen_existence: set[tuple[str, str, str]] = set()
        for c in constraints:
            t, nt, rt, pn = _constraint_identity(c)
            if t in (GraphConstraintType.EXISTENCE.value, "EXISTENCE"):
                seen_existence.add((nt or "", rt or "", pn))

        for node in data.get("node_types") or []:
            if not isinstance(node, dict):
                continue
            label = node.get("label")
            if not label:
                continue
            for prop in node.get("properties") or []:
                if not isinstance(prop, dict):
                    continue
                if prop.get("required") is not True:
                    continue
                pname = prop.get("name")
                if not pname:
                    continue
                key = (label, "", pname)
                if key in seen_existence:
                    prop["required"] = False
                    continue
                constraints.append(
                    {
                        "type": GraphConstraintType.EXISTENCE.value,
                        "node_type": label,
                        "property_name": pname,
                        "relationship_type": None,
                    }
                )
                seen_existence.add(key)
                prop["required"] = False

        node_types_in = data.get("node_types") or []
        if node_types_in:
            node_types_list = list(node_types_in)
            for i, node in enumerate(node_types_list):
                if not isinstance(node, NodeType):
                    continue
                label = node.label
                if not label:
                    continue
                new_props: list[PropertyType] = []
                for prop in node.properties:
                    if not _property_type_required_is_true(prop):
                        new_props.append(prop)
                        continue
                    pname = prop.name
                    if not pname:
                        new_props.append(prop)
                        continue
                    key = (label, "", pname)
                    if key in seen_existence:
                        new_props.append(prop.model_copy(update={"required": False}))
                        continue
                    constraints.append(
                        {
                            "type": GraphConstraintType.EXISTENCE.value,
                            "node_type": label,
                            "property_name": pname,
                            "relationship_type": None,
                        }
                    )
                    seen_existence.add(key)
                    new_props.append(prop.model_copy(update={"required": False}))
                node_types_list[i] = node.model_copy(update={"properties": new_props})
            data["node_types"] = tuple(node_types_list)

        for rel in data.get("relationship_types") or []:
            if not isinstance(rel, dict):
                continue
            rlabel = rel.get("label")
            if not rlabel:
                continue
            for prop in rel.get("properties") or []:
                if not isinstance(prop, dict):
                    continue
                if prop.get("required") is not True:
                    continue
                pname = prop.get("name")
                if not pname:
                    continue
                key = ("", rlabel, pname)
                if key in seen_existence:
                    prop["required"] = False
                    continue
                constraints.append(
                    {
                        "type": GraphConstraintType.EXISTENCE.value,
                        "node_type": "",
                        "property_name": pname,
                        "relationship_type": rlabel,
                    }
                )
                seen_existence.add(key)
                prop["required"] = False

        rel_types_in = data.get("relationship_types") or []
        if rel_types_in:
            rel_types_list = list(rel_types_in)
            for i, rel in enumerate(rel_types_list):
                if not isinstance(rel, RelationshipType):
                    continue
                rlabel = rel.label
                if not rlabel:
                    continue
                rel_new_props: list[PropertyType] = []
                for prop in rel.properties:
                    if not _property_type_required_is_true(prop):
                        rel_new_props.append(prop)
                        continue
                    pname = prop.name
                    if not pname:
                        rel_new_props.append(prop)
                        continue
                    key = ("", rlabel, pname)
                    if key in seen_existence:
                        rel_new_props.append(
                            prop.model_copy(update={"required": False})
                        )
                        continue
                    constraints.append(
                        {
                            "type": GraphConstraintType.EXISTENCE.value,
                            "node_type": "",
                            "property_name": pname,
                            "relationship_type": rlabel,
                        }
                    )
                    seen_existence.add(key)
                    rel_new_props.append(prop.model_copy(update={"required": False}))
                rel_types_list[i] = rel.model_copy(update={"properties": rel_new_props})
            data["relationship_types"] = tuple(rel_types_list)

        data["constraints"] = tuple(constraints)
        return data

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
            ctype = constraint.type
            if isinstance(ctype, str):
                ctype = GraphConstraintType(ctype)

            if ctype == GraphConstraintType.UNIQUENESS:
                if constraint.node_type not in self._node_type_index:
                    raise SchemaValidationError(
                        f"Constraint references undefined node type: {constraint.node_type}"
                    )
                node_type = self._node_type_index[constraint.node_type]
                valid_property_names = {p.name for p in node_type.properties}
                if constraint.property_name not in valid_property_names:
                    raise SchemaValidationError(
                        f"Constraint references undefined property '{constraint.property_name}' "
                        f"on node type '{constraint.node_type}'. "
                        f"Valid properties: {valid_property_names}"
                    )
            elif ctype in (
                GraphConstraintType.EXISTENCE,
                GraphConstraintType.KEY,
            ):
                kind: Literal["EXISTENCE", "KEY"] = (
                    "EXISTENCE" if ctype == GraphConstraintType.EXISTENCE else "KEY"
                )
                _validate_existence_or_key_property_defined(
                    constraint,
                    node_type_index=self._node_type_index,
                    relationship_type_index=self._relationship_type_index,
                    kind=kind,
                )
        return self

    @model_validator(mode="after")
    def validate_constraints_uniqueness_key_exclusivity(self) -> Self:
        """Neo4j: property uniqueness and key constraints cannot share the same schema."""
        node_uniqueness: set[tuple[str, str]] = set()
        node_key: set[tuple[str, str]] = set()
        for c in self.constraints:
            ct = c.type
            if isinstance(ct, str):
                ct = GraphConstraintType(ct)
            if ct == GraphConstraintType.UNIQUENESS:
                node_uniqueness.add((c.node_type, c.property_name))
            elif ct == GraphConstraintType.KEY:
                if (
                    c.node_type
                    and str(c.node_type).strip()
                    and not (c.relationship_type and str(c.relationship_type).strip())
                ):
                    node_key.add((c.node_type, c.property_name))
        overlap = node_uniqueness & node_key
        if overlap:
            nt, pn = next(iter(overlap))
            raise SchemaValidationError(
                f"UNIQUENESS and KEY constraints cannot apply to the same node type and property "
                f"({nt!r}, {pn!r})."
            )
        return self

    def node_type_from_label(self, label: str) -> Optional[NodeType]:
        return self._node_type_index.get(label)

    def relationship_type_from_label(self, label: str) -> Optional[RelationshipType]:
        return self._relationship_type_index.get(label)

    def existence_property_names_for_node(self, label: str) -> set[str]:
        """Property names that have an EXISTENCE constraint for this node label."""
        names: set[str] = set()
        for c in self.constraints:
            ct = c.type
            if isinstance(ct, str):
                ct = GraphConstraintType(ct)
            if ct != GraphConstraintType.EXISTENCE:
                continue
            if c.node_type == label and not (
                c.relationship_type and str(c.relationship_type).strip()
            ):
                names.add(c.property_name)
        return names

    def existence_property_names_for_relationship(self, rel_label: str) -> set[str]:
        """Property names that have an EXISTENCE constraint for this relationship type."""
        names: set[str] = set()
        for c in self.constraints:
            ct = c.type
            if isinstance(ct, str):
                ct = GraphConstraintType(ct)
            if ct != GraphConstraintType.EXISTENCE:
                continue
            if c.relationship_type == rel_label:
                names.add(c.property_name)
        return names

    def key_property_names_for_node(self, label: str) -> set[str]:
        """Property names under a KEY constraint for this node label (Neo4j NODE KEY)."""
        names: set[str] = set()
        for c in self.constraints:
            ct = c.type
            if isinstance(ct, str):
                ct = GraphConstraintType(ct)
            if ct != GraphConstraintType.KEY:
                continue
            if c.node_type == label and not (
                c.relationship_type and str(c.relationship_type).strip()
            ):
                names.add(c.property_name)
        return names

    def key_property_names_for_relationship(self, rel_label: str) -> set[str]:
        """Property names under a KEY constraint for this relationship type (Neo4j RELATIONSHIP KEY)."""
        names: set[str] = set()
        for c in self.constraints:
            ct = c.type
            if isinstance(ct, str):
                ct = GraphConstraintType(ct)
            if ct != GraphConstraintType.KEY:
                continue
            if c.relationship_type == rel_label:
                names.add(c.property_name)
        return names

    def uniqueness_property_names_for_node(self, label: str) -> set[str]:
        """Property names under a UNIQUENESS constraint for this node label."""
        names: set[str] = set()
        for c in self.constraints:
            ct = c.type
            if isinstance(ct, str):
                ct = GraphConstraintType(ct)
            if ct != GraphConstraintType.UNIQUENESS:
                continue
            if c.node_type == label:
                names.add(c.property_name)
        return names

    def mandatory_property_names_for_node(self, label: str) -> set[str]:
        """Property names required to be present (non-null): EXISTENCE or KEY constraints."""
        return self.existence_property_names_for_node(
            label
        ) | self.key_property_names_for_node(label)

    def mandatory_property_names_for_relationship(self, rel_label: str) -> set[str]:
        """Property names required to be present (non-null): EXISTENCE or KEY constraints."""
        return self.existence_property_names_for_relationship(
            rel_label
        ) | self.key_property_names_for_relationship(rel_label)

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Override for structured output compatibility.

        OpenAI requires:
        - additionalProperties: false on all objects
        - All properties must be in required array

        VertexAI requires:
        - No 'const' keyword (convert to enum with single value)

        Prefer :class:`~neo4j_graphrag.experimental.components.graph_schema_extraction.GraphSchemaExtractionOutput`
        for schema-from-text structured output (leaner schema).
        """
        schema = super().model_json_schema(**kwargs)
        make_strict_json_schema_for_structured_output(schema)
        return schema

    @classmethod
    def from_extraction_output(cls, dto: GraphSchemaExtractionOutput) -> Self:
        """Build a :class:`GraphSchema` from :class:`GraphSchemaExtractionOutput`.

        Applies the same cross-reference filtering and validation as
        :class:`SchemaFromTextExtractor`.
        """
        from neo4j_graphrag.experimental.components.graph_schema_extraction import (
            GraphSchemaExtractionOutput,
            wire_extraction_constraints_for_graph_schema,
        )

        if not isinstance(dto, GraphSchemaExtractionOutput):
            raise TypeError(
                f"Expected GraphSchemaExtractionOutput, got {type(dto).__name__}"
            )
        payload = dto.model_dump(mode="python")
        payload["constraints"] = wire_extraction_constraints_for_graph_schema(
            payload.get("constraints") or []
        )
        return cast(Self, validate_extraction_dict_to_graph_schema(payload))

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
        patterns: Optional[Sequence[Union[Tuple[str, str, str], Pattern]]] = None,
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
        patterns: Optional[Sequence[Union[Tuple[str, str, str], Pattern]]] = None,
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


def _extraction_filter_invalid_patterns(
    patterns: Any,
    node_types: List[Dict[str, Any]],
    relationship_types: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """Filter out patterns that reference undefined node or relationship types."""
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

    valid_node_labels = {node_type["label"] for node_type in node_types}
    valid_relationship_labels = {rel_type["label"] for rel_type in relationship_types}

    filtered_patterns = []
    for pattern in patterns:
        if isinstance(pattern, dict):
            if not all(k in pattern for k in ("source", "relationship", "target")):
                continue
            entity1 = pattern["source"]
            relation = pattern["relationship"]
            entity2 = pattern["target"]
        elif isinstance(pattern, (list, tuple)):
            if len(pattern) != 3:
                continue
            entity1, relation, entity2 = pattern
        elif isinstance(pattern, Pattern):
            entity1, relation, entity2 = pattern
        else:
            continue

        if (
            entity1 in valid_node_labels
            and entity2 in valid_node_labels
            and relation in valid_relationship_labels
        ):
            filtered_patterns.append(pattern)
        else:
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


def _extraction_filter_invalid_constraints(
    constraints: List[Dict[str, Any]],
    node_types: List[Dict[str, Any]],
    relationship_types: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Filter constraints that reference unknown types or invalid properties."""
    if not constraints:
        return []

    node_type_properties: Dict[str, set[str]] = {}
    for node_type_dict in node_types:
        label = node_type_dict.get("label")
        if label:
            properties = node_type_dict.get("properties", [])
            property_names = {p.get("name") for p in properties if p.get("name")}
            node_type_properties[label] = property_names

    rel_type_properties: Dict[str, set[str]] = {}
    for rel_dict in relationship_types or []:
        label = rel_dict.get("label")
        if label:
            properties = rel_dict.get("properties", [])
            property_names = {p.get("name") for p in properties if p.get("name")}
            rel_type_properties[label] = property_names

    valid_node_labels = set(node_type_properties.keys())
    valid_rel_labels = set(rel_type_properties.keys())

    filtered_constraints = []
    for constraint in constraints:
        ctype = constraint.get("type")
        if ctype not in (
            GraphConstraintType.UNIQUENESS.value,
            GraphConstraintType.EXISTENCE.value,
            GraphConstraintType.KEY.value,
        ):
            logging.info(
                f"Filtering out constraint: {constraint}. "
                f"Unsupported constraint type (expected UNIQUENESS, EXISTENCE, or KEY)."
            )
            continue

        if not constraint.get("property_name"):
            logging.info(
                f"Filtering out constraint: {constraint}. "
                f"Property name is not provided."
            )
            continue

        if ctype == GraphConstraintType.UNIQUENESS.value:
            if not node_types:
                logging.info(
                    f"Filtering out constraint: {constraint}. "
                    f"No node types are defined."
                )
                continue
            node_type = constraint.get("node_type")
            if not node_type or node_type not in valid_node_labels:
                logging.info(
                    f"Filtering out constraint: {constraint}. "
                    f"Node type '{node_type}' is not valid. Valid node types: {valid_node_labels}"
                )
                continue
            property_name = constraint.get("property_name")
            if property_name not in node_type_properties.get(node_type, set()):
                logging.info(
                    f"Filtering out constraint: {constraint}. "
                    f"Property '{property_name}' does not exist on node type '{node_type}'. "
                    f"Valid properties: {node_type_properties.get(node_type, set())}"
                )
                continue
            filtered_constraints.append(constraint)
            continue

        # EXISTENCE or KEY
        node_type = constraint.get("node_type") or ""
        rel_type = constraint.get("relationship_type")
        has_node = bool(str(node_type).strip())
        has_rel = bool(rel_type and str(rel_type).strip())
        if has_node == has_rel:
            logging.info(
                f"Filtering out constraint: {constraint}. "
                f"{ctype} requires exactly one of node_type or relationship_type."
            )
            continue
        if has_node:
            if node_type not in valid_node_labels:
                logging.info(
                    f"Filtering out constraint: {constraint}. "
                    f"Node type '{node_type}' is not valid. Valid node types: {valid_node_labels}"
                )
                continue
            property_name = constraint.get("property_name")
            if property_name not in node_type_properties.get(node_type, set()):
                logging.info(
                    f"Filtering out constraint: {constraint}. "
                    f"Property '{property_name}' does not exist on node type '{node_type}'. "
                    f"Valid properties: {node_type_properties.get(node_type, set())}"
                )
                continue
            filtered_constraints.append(constraint)
        else:
            assert rel_type is not None
            if rel_type not in valid_rel_labels:
                logging.info(
                    f"Filtering out constraint: {constraint}. "
                    f"Relationship type '{rel_type}' is not valid. "
                    f"Valid relationship types: {valid_rel_labels}"
                )
                continue
            property_name = constraint.get("property_name")
            if property_name not in rel_type_properties.get(rel_type, set()):
                logging.info(
                    f"Filtering out constraint: {constraint}. "
                    f"Property '{property_name}' does not exist on relationship type '{rel_type}'. "
                    f"Valid properties: {rel_type_properties.get(rel_type, set())}"
                )
                continue
            filtered_constraints.append(constraint)

    return filtered_constraints


def _extraction_apply_cross_reference_filters(
    extracted_node_types: List[Dict[str, Any]],
    extracted_relationship_types: Optional[List[Dict[str, Any]]],
    extracted_patterns: Any,
    extracted_constraints: Optional[List[Dict[str, Any]]],
) -> tuple[Any, Optional[List[Dict[str, Any]]]]:
    if extracted_patterns:
        extracted_patterns = _extraction_filter_invalid_patterns(
            extracted_patterns,
            extracted_node_types,
            extracted_relationship_types,
        )

    if extracted_constraints:
        extracted_constraints = _extraction_filter_invalid_constraints(
            extracted_constraints,
            extracted_node_types,
            extracted_relationship_types,
        )

    return extracted_patterns, extracted_constraints


def validate_extraction_dict_to_graph_schema(
    extracted_schema: Dict[str, Any],
) -> GraphSchema:
    """Cross-reference filter and build :class:`GraphSchema` from an extraction dict.

    Used by :meth:`GraphSchema.from_extraction_output` and
    :class:`SchemaFromTextExtractor` (V1 and V2). Does not require a configured LLM.
    """
    node_types = extracted_schema.get("node_types") or []
    rel_types = extracted_schema.get("relationship_types")
    patterns = extracted_schema.get("patterns")
    constraints = extracted_schema.get("constraints")

    patterns, constraints = _extraction_apply_cross_reference_filters(
        node_types, rel_types, patterns, constraints
    )

    try:
        schema = GraphSchema.model_validate(
            {
                "node_types": node_types,
                "relationship_types": rel_types,
                "patterns": patterns,
                "constraints": constraints or [],
            }
        )
        logger.debug(f"Extracted schema: {schema}")
        return schema
    except ValidationError as e:
        raise SchemaExtractionError(
            f"LLM response does not conform to GraphSchema: {str(e)}"
        ) from e


class SchemaFromTextExtractor(BaseSchemaBuilder):
    """
    A component for constructing GraphSchema objects from the output of an LLM after
    automatic schema extraction from text.

    Args:
        llm (LLMInterface): The language model to use for schema extraction.
        prompt_template (Optional[PromptTemplate]): A custom prompt template to use for extraction.
        llm_params (Optional[Dict[str, Any]]): Additional parameters passed to the LLM.
        use_structured_output (bool): Whether to use structured output (LLMInterfaceV2) with
            :class:`~neo4j_graphrag.experimental.components.graph_schema_extraction.GraphSchemaExtractionOutput`.
            Only supported for OpenAILLM and VertexAILLM. Defaults to False (uses V1 prompt-based JSON extraction).

    Example with V1 (default, prompt-based JSON):

    .. code-block:: python

        from neo4j_graphrag.experimental.components.schema import SchemaFromTextExtractor
        from neo4j_graphrag.llm import OpenAILLM

        llm = OpenAILLM(model_name="gpt-5", model_params={"temperature": 0})
        extractor = SchemaFromTextExtractor(llm=llm)

    Example with V2 (structured output):

    .. code-block:: python

        from neo4j_graphrag.experimental.components.schema import SchemaFromTextExtractor
        from neo4j_graphrag.llm import OpenAILLM

        llm = OpenAILLM(model_name="gpt-5")
        extractor = SchemaFromTextExtractor(llm=llm, use_structured_output=True)
    """

    def __init__(
        self,
        llm: LLMInterface,
        prompt_template: Optional[PromptTemplate] = None,
        llm_params: Optional[Dict[str, Any]] = None,
        use_structured_output: bool = False,
    ) -> None:
        self._llm: LLMInterface = llm
        self._prompt_template: PromptTemplate = (
            prompt_template or SchemaExtractionTemplate()
        )
        self._llm_params: dict[str, Any] = llm_params or {}
        self.use_structured_output = use_structured_output

        # Validate that structured output is only used with supported LLMs
        if use_structured_output and not llm.supports_structured_output:
            raise ValueError(
                f"Structured output is not supported by {type(llm).__name__}. "
                f"Please use a model that supports structured output, or set use_structured_output=False."
            )

    def _filter_items_without_labels(
        self, items: List[Dict[str, Any]], item_type: str
    ) -> List[Dict[str, Any]]:
        """Filter out items that have no valid labels."""
        filtered_items = []
        for item in items:
            if isinstance(item, str):
                if item and " " not in item and not item.startswith("{"):
                    # Add default property for node types to satisfy min_length=1 constraint
                    # This matches the behavior of NodeType.validate_input_if_string
                    if item_type == "node type":
                        filtered_items.append(
                            {
                                "label": item,
                                "properties": [{"name": "name", "type": "STRING"}],
                                "additional_properties": True,
                            }
                        )
                    else:
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

    def _clean_json_content(self, content: str) -> str:
        content = content.strip()

        # Remove markdown code block markers if present
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
        content = re.sub(r"```\s*$", "", content, flags=re.MULTILINE)

        return content.strip()

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response into schema dictionary.

        Args:
            content: JSON string from LLM response

        Returns:
            Parsed dictionary

        Raises:
            SchemaExtractionError: If content is not valid JSON
        """
        try:
            result = json.loads(content)
            return cast(Dict[str, Any], result)
        except json.JSONDecodeError as exc:
            raise SchemaExtractionError("LLM response is not valid JSON.") from exc

    def _parse_and_normalize_schema(self, content: str) -> Dict[str, Any]:
        """Parse and normalize V1 schema response (handles lists/dicts).

        V1 (prompt-based) extraction sometimes returns lists instead of dicts.
        This method normalizes the response to always return a dict.

        Args:
            content: JSON string from LLM response

        Returns:
            Normalized schema dictionary

        Raises:
            SchemaExtractionError: If content is not valid JSON or has unexpected format
        """
        extracted_schema = self._parse_llm_response(content)

        # Handle list responses
        if isinstance(extracted_schema, list):
            if len(extracted_schema) == 0:
                logging.info(
                    "LLM returned an empty list for schema. Falling back to empty schema."
                )
                return {}
            elif isinstance(extracted_schema[0], dict):
                return extracted_schema[0]
            else:
                raise SchemaExtractionError(
                    f"Expected a dictionary or list of dictionaries, but got list containing: {type(extracted_schema[0])}"
                )
        elif isinstance(extracted_schema, dict):
            return extracted_schema
        else:
            raise SchemaExtractionError(
                f"Unexpected schema format returned from LLM: {type(extracted_schema)}. Expected a dictionary or list of dictionaries."
            )

    def _apply_v1_filters(self, extracted_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Apply V1-specific filters before cross-reference filtering.

        V1 (prompt-based) extraction requires additional filtering:
        - Remove nodes/relationships without labels
        - Remove nodes with no properties (after property filtering)

        Legacy ``required`` on properties is normalized by :class:`GraphSchema` validation
        (migration to ``EXISTENCE`` constraints). Invalid ``required`` values fail validation
        or are handled by Pydantic coercion where applicable.

        Args:
            extracted_schema: Raw schema dictionary from LLM

        Returns:
            Filtered schema dictionary
        """
        node_types = extracted_schema.get("node_types") or []
        rel_types = extracted_schema.get("relationship_types")

        # Filter items without labels
        node_types = self._filter_nodes_without_labels(node_types)
        if rel_types:
            rel_types = self._filter_relationships_without_labels(rel_types)

        # Filter nodes with no properties (after property filtering)
        # This prevents validation errors from min_length=1 constraint on NodeType.properties
        nodes_before = len(node_types)
        node_types = [
            node for node in node_types if len(node.get("properties", [])) > 0
        ]
        if len(node_types) < nodes_before:
            removed_count = nodes_before - len(node_types)
            logging.info(
                f"Filtered out {removed_count} node type(s) with no properties after property validation."
            )

        extracted_schema["node_types"] = node_types
        extracted_schema["relationship_types"] = rel_types
        return extracted_schema

    async def _run_with_structured_output(self, prompt: str) -> GraphSchema:
        """Extract schema using structured output (V2).

        V2 uses LLMInterfaceV2 with
        :class:`~neo4j_graphrag.experimental.components.graph_schema_extraction.GraphSchemaExtractionOutput`
        as ``response_format``, then converts to :class:`GraphSchema`. Requires OpenAI or VertexAI.

        Args:
            prompt: Formatted prompt for schema extraction

        Returns:
            Validated GraphSchema object

        Raises:
            RuntimeError: If LLM is not OpenAILLM or VertexAILLM
            SchemaExtractionError: If LLM generation or validation fails
        """
        from neo4j_graphrag.experimental.components.graph_schema_extraction import (
            GraphSchemaExtractionOutput,
        )

        # Capability check
        # This should never happen due to __init__ validation
        if not self._llm.supports_structured_output:
            raise RuntimeError(
                f"Structured output is not supported by {type(self._llm).__name__}"
            )

        # Invoke LLM with structured output
        messages = [LLMMessage(role="user", content=prompt)]
        try:
            llm_result = await self._llm.ainvoke(
                messages,  # type: ignore[arg-type]
                response_format=GraphSchemaExtractionOutput,  # type: ignore[call-arg]
            )
        except LLMGenerationError as e:
            raise SchemaExtractionError("Failed to generate schema from text") from e

        try:
            dto = GraphSchemaExtractionOutput.model_validate(
                self._parse_llm_response(llm_result.content)
            )
        except ValidationError as e:
            raise SchemaExtractionError(
                "LLM response does not conform to GraphSchemaExtractionOutput."
            ) from e

        return GraphSchema.from_extraction_output(dto)

    async def _run_with_prompt_based_extraction(self, prompt: str) -> GraphSchema:
        """Extract schema using prompt-based JSON extraction (V1).

        V1 uses standard LLM prompting with JSON output. This requires additional
        filtering and cleanup compared to V2 structured output.

        Args:
            prompt: Formatted prompt for schema extraction

        Returns:
            Validated GraphSchema object

        Raises:
            LLMGenerationError: If LLM generation fails
            SchemaExtractionError: If parsing or validation fails
        """
        # Invoke LLM
        try:
            response = await self._llm.ainvoke(prompt, **self._llm_params)
            content = response.content
        except LLMGenerationError as e:
            raise LLMGenerationError("Failed to generate schema from text") from e

        # Clean and parse response
        content = self._clean_json_content(content)
        extracted_schema = self._parse_and_normalize_schema(content)

        # Apply V1-specific filtering
        extracted_schema = self._apply_v1_filters(extracted_schema)

        return validate_extraction_dict_to_graph_schema(extracted_schema)

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

        if self.use_structured_output:
            return await self._run_with_structured_output(prompt)
        else:
            return await self._run_with_prompt_based_extraction(prompt)


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
    def _extract_graph_constraints_from_metadata(
        structured_schema: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build EXISTENCE / KEY constraint dicts from Neo4j ``SHOW CONSTRAINTS`` metadata."""
        schema_metadata = structured_schema.get("metadata", {})
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str, str]] = set()

        for constraint in schema_metadata.get("constraint", []):
            ctype = constraint.get("type")
            properties = constraint.get("properties") or []
            labels = constraint.get("labelsOrTypes") or []
            if not properties or not labels:
                continue
            prop = properties[0]
            lab = labels[0]

            if ctype == "NODE_PROPERTY_EXISTENCE":
                dedupe_key = ("EXISTENCE", "NODE", lab, "", prop)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                result.append(
                    {
                        "type": GraphConstraintType.EXISTENCE.value,
                        "node_type": lab,
                        "property_name": prop,
                        "relationship_type": None,
                    }
                )
            elif ctype == "NODE_KEY":
                dedupe_key = ("KEY", "NODE", lab, "", prop)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                result.append(
                    {
                        "type": GraphConstraintType.KEY.value,
                        "node_type": lab,
                        "property_name": prop,
                        "relationship_type": None,
                    }
                )
            elif ctype == "RELATIONSHIP_PROPERTY_EXISTENCE":
                dedupe_key = ("EXISTENCE", "REL", "", lab, prop)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                result.append(
                    {
                        "type": GraphConstraintType.EXISTENCE.value,
                        "node_type": "",
                        "property_name": prop,
                        "relationship_type": lab,
                    }
                )
            elif ctype == "RELATIONSHIP_KEY":
                dedupe_key = ("KEY", "REL", "", lab, prop)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                result.append(
                    {
                        "type": GraphConstraintType.KEY.value,
                        "node_type": "",
                        "property_name": prop,
                        "relationship_type": lab,
                    }
                )

        return result

    def _to_schema_entity_dict(
        self,
        key: str,
        property_dict: list[dict[str, Any]],
    ) -> dict[str, Any]:
        entity_dict: dict[str, Any] = {
            "label": key,
            "properties": [
                {
                    "name": p["property"],
                    "type": p["type"],
                }
                for p in property_dict
            ],
        }
        if self.additional_properties:
            entity_dict["additional_properties"] = self.additional_properties
        return entity_dict

    async def run(self, *args: Any, **kwargs: Any) -> GraphSchema:
        structured_schema = get_structured_schema(self.driver, database=self.database)
        graph_constraints = self._extract_graph_constraints_from_metadata(
            structured_schema
        )

        # node label with properties
        node_labels = set(structured_schema["node_props"].keys())
        node_types = [
            self._to_schema_entity_dict(key, properties)
            for key, properties in structured_schema["node_props"].items()
        ]

        # relationships with properties
        rel_labels = set(structured_schema["rel_props"].keys())
        relationship_types = [
            self._to_schema_entity_dict(key, properties)
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
            "constraints": graph_constraints,
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
