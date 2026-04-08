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

"""Contract tests: keep :class:`GraphSchemaExtractionOutput` aligned with :class:`GraphSchema` / runtime models.

If these fail after a refactor, update the extraction models **and** conversion together.
"""

from __future__ import annotations

from neo4j_graphrag.experimental.components.graph_schema_extraction import (
    ExtractedNodeType,
    ExtractedPropertyType,
    ExtractedRelationshipType,
    GraphSchemaExtractionOutput,
)
from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    Neo4jPropertyTypeName,
    NodeType,
    PropertyType,
    RelationshipType,
)


def test_extracted_property_type_field_names_match_property_type() -> None:
    """Extraction wire format must stay in sync with :class:`PropertyType` for mapped fields."""
    assert set(ExtractedPropertyType.model_fields) == set(PropertyType.model_fields)


def test_extracted_property_type_uses_same_type_annotation_as_property_type() -> None:
    assert (
        ExtractedPropertyType.model_fields["type"].annotation
        is PropertyType.model_fields["type"].annotation
    )
    assert (
        ExtractedPropertyType.model_fields["type"].annotation is Neo4jPropertyTypeName
    )


def test_extracted_node_type_has_core_node_type_fields_only() -> None:
    """Lean model: same core keys as :class:`NodeType` except ``additional_properties`` (set at conversion)."""
    assert set(ExtractedNodeType.model_fields) == {"label", "description", "properties"}
    assert set(ExtractedNodeType.model_fields).issubset(set(NodeType.model_fields))


def test_extracted_relationship_type_has_core_relationship_type_fields_only() -> None:
    assert set(ExtractedRelationshipType.model_fields) == {
        "label",
        "description",
        "properties",
    }
    assert set(ExtractedRelationshipType.model_fields).issubset(
        set(RelationshipType.model_fields)
    )


def test_graph_schema_extraction_output_root_keys_match_validate_payload() -> None:
    """Keys must match the dict passed to :meth:`GraphSchema.model_validate` from extraction (no ``additional_*``)."""
    assert set(GraphSchemaExtractionOutput.model_fields) == {
        "node_types",
        "relationship_types",
        "patterns",
        "constraints",
    }


def test_graph_schema_model_fields_contain_extraction_superset() -> None:
    """Runtime schema adds pipeline-only fields; extraction is a strict subset at the root."""
    extraction_keys = set(GraphSchemaExtractionOutput.model_fields)
    graph_keys = set(GraphSchema.model_fields)
    assert extraction_keys.issubset(graph_keys)
    assert {
        "additional_node_types",
        "additional_relationship_types",
        "additional_patterns",
    }.issubset(graph_keys)
