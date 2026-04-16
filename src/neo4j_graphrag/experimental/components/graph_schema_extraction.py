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

"""Pydantic models for schema-from-text **structured output** only.

These types are a lean wire format for ``response_format`` with supported LLMs.
Pipeline code uses :class:`~neo4j_graphrag.experimental.components.schema.GraphSchema`
exclusively; convert via :meth:`~neo4j_graphrag.experimental.components.schema.GraphSchema.from_extraction_output`.

Vertex AI builds JSON Schema from Pydantic and parses it with protobuf; it rejects
``anyOf`` branches that use JSON Schema ``type: "null"`` (e.g. ``Optional[str]``).
Extraction-specific models therefore avoid nullable fields (e.g. use empty string
sentinels where runtime :class:`~neo4j_graphrag.experimental.components.schema.ConstraintType`
uses ``None``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neo4j_graphrag.experimental.components.schema import (
    Neo4jPropertyTypeName,
    Pattern,
)
from neo4j_graphrag.utils.json_schema_structured_output import (
    make_strict_json_schema_for_structured_output,
)


class ExtractedPropertyType(BaseModel):
    """Property definition aligned with :class:`~neo4j_graphrag.experimental.components.schema.PropertyType` for extraction."""

    name: str
    type: Neo4jPropertyTypeName
    description: str = ""
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExtractedNodeType(BaseModel):
    """Node type for extraction structured output (no ``additional_properties``; set at conversion)."""

    label: str
    description: str = ""
    properties: list[ExtractedPropertyType] = Field(min_length=1)
    model_config = ConfigDict(extra="forbid")


class ExtractedRelationshipType(BaseModel):
    """Relationship type for extraction structured output."""

    label: str
    description: str = ""
    properties: list[ExtractedPropertyType] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class ExtractedConstraintType(BaseModel):
    """Constraint in schema-from-text structured output (Vertex/OpenAI JSON Schema).

    This is a **wire DTO** only: shapes the JSON Schema (e.g. no nullable ``relationship_type``,
    which Vertex's protobuf parser rejects). Semantic rules match
    :class:`~neo4j_graphrag.experimental.components.schema.ConstraintType`; those are enforced
    when building :class:`~neo4j_graphrag.experimental.components.schema.GraphSchema`, not here.

    ``property_names`` is the canonical field (list of one or more property names).
    ``property_name`` is accepted for backward compatibility and auto-migrated.
    """

    type: Literal["UNIQUENESS", "EXISTENCE", "KEY"]
    property_name: str = ""
    property_names: list[str] = Field(default_factory=list)
    node_type: str = ""
    relationship_type: str = ""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def migrate_property_name_to_property_names(cls, data: Any) -> Any:
        """Backward-compat: ``property_name`` → ``property_names``."""
        if not isinstance(data, dict):
            return data
        pn = data.get("property_name") or ""
        pns = data.get("property_names") or []
        if isinstance(pns, str):
            pns = [pns]
        else:
            pns = list(pns)
        if pns and not pn:
            data["property_name"] = pns[0]
        elif pn and not pns:
            data["property_names"] = [pn]
        elif pn and pns:
            data["property_name"] = pns[0]
        return data


def wire_extraction_constraints_for_graph_schema(
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map extraction constraint dicts to values compatible with :class:`~neo4j_graphrag.experimental.components.schema.ConstraintType`.

    Empty ``relationship_type`` (the wire \"unset\" sentinel) becomes ``None`` for runtime validation.
    ``property_names`` is propagated from the extraction DTO.
    """
    out: list[dict[str, Any]] = []
    for c in constraints:
        d = dict(c)
        rt = d.get("relationship_type")
        if rt is None or (isinstance(rt, str) and rt.strip() == ""):
            d["relationship_type"] = None
        # Ensure property_names is propagated
        pns = d.get("property_names") or []
        pn = d.get("property_name") or ""
        if not pns and pn:
            d["property_names"] = [pn]
        elif pns and not pn:
            d["property_name"] = pns[0]
        out.append(d)
    return out


class GraphSchemaExtractionOutput(BaseModel):
    """JSON shape for LLM schema-from-text structured output (V2).

    Convert to :class:`~neo4j_graphrag.experimental.components.schema.GraphSchema` with
    :meth:`~neo4j_graphrag.experimental.components.schema.GraphSchema.from_extraction_output`.
    """

    node_types: list[ExtractedNodeType] = Field(default_factory=list)
    relationship_types: list[ExtractedRelationshipType] = Field(default_factory=list)
    patterns: list[Pattern] = Field(default_factory=list)
    constraints: list[ExtractedConstraintType] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """JSON Schema compatible with OpenAI / Vertex structured JSON output."""
        schema = super().model_json_schema(**kwargs)
        make_strict_json_schema_for_structured_output(schema)
        return schema
