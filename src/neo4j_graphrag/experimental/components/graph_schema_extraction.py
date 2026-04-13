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
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from neo4j_graphrag.experimental.components.schema import (
    ConstraintType,
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


class GraphSchemaExtractionOutput(BaseModel):
    """JSON shape for LLM schema-from-text structured output (V2).

    Convert to :class:`~neo4j_graphrag.experimental.components.schema.GraphSchema` with
    :meth:`~neo4j_graphrag.experimental.components.schema.GraphSchema.from_extraction_output`.
    """

    node_types: list[ExtractedNodeType] = Field(default_factory=list)
    relationship_types: list[ExtractedRelationshipType] = Field(default_factory=list)
    patterns: list[Pattern] = Field(default_factory=list)
    constraints: list[ConstraintType] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """JSON Schema compatible with OpenAI / Vertex structured JSON output."""
        schema = super().model_json_schema(**kwargs)
        make_strict_json_schema_for_structured_output(schema)
        return schema
