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
"""Format Neo4j graph data as Parquet files (per-label nodes, per-type relationships).

Parquet filenames are derived from node labels and relationship types and are
sanitized for filesystem and Neo4j import compatibility (safe characters:
[a-zA-Z0-9_], Unicode normalized/transliterated to ASCII).
"""

from __future__ import annotations

import logging
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, DefaultDict, Optional

from neo4j_graphrag.experimental.components.types import (
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)

logger = logging.getLogger(__name__)

_FALLBACK_FILESTEM = "unnamed"


def _is_allowed_filestem_char(c: str) -> bool:
    """True if c is in [a-zA-Z0-9_]."""
    return c.isalnum() or c == "_"


def sanitize_parquet_filestem(name: str) -> str:
    """Convert a label or name into a safe Parquet filename stem.

    Safe means: only [a-zA-Z0-9_]. Accented and other Unicode characters
    are transliterated to ASCII (e.g. Ü -> U, é -> e). Disallowed
    characters are replaced with underscore. If the result is empty,
    returns a fallback stem.

    Used when building Parquet filenames from node labels and relationship
    type names for filesystem and Neo4j Aura KG import compatibility.

    Args:
        name: Original name (e.g. node label, relationship type name).

    Returns:
        A string safe for use as the stem of a parquet filename.
    """
    if not name:
        return _FALLBACK_FILESTEM
    # Transliterate to ASCII (NFKD decomposes e.g. é -> e + combining accent)
    normalized = unicodedata.normalize("NFKD", name)
    ascii_bytes = normalized.encode("ascii", errors="ignore")
    stem = ascii_bytes.decode("ascii")
    # Replace any remaining disallowed characters with underscore
    result = "".join(c if _is_allowed_filestem_char(c) else "_" for c in stem)
    if not result:
        return _FALLBACK_FILESTEM
    return result


def get_unique_properties_for_node_type(
    schema: Optional[dict[str, Any]], node_label: str
) -> list[str]:
    """Extract unique property names from schema constraints for a given node type.

    1. If the schema has constraints, use uniqueness constraints
    2. Otherwise, fall back to "__id__"

    Args:
        schema: The GraphSchema as a dictionary (may contain 'constraints' key)
        node_label: The label for the node type

    Returns:
        List of property names that have uniqueness constraints
    """
    default = ["__id__"]
    if not schema:
        return default

    constraints = schema.get("constraints", ())
    unique_properties: list[str] = []

    for constraint in constraints:
        # Check if this constraint applies to the node's label
        if constraint.get("type") == "UNIQUENESS":
            constraint_node_type = constraint.get("node_type", "")
            if constraint_node_type == node_label:
                property_name = constraint.get("property_name", "")
                if property_name:
                    unique_properties.append(property_name)

    return unique_properties or default


@dataclass
class FileMetadata:
    """Metadata about a generated Parquet file."""

    filename: str
    schema: Any  # pa.Schema - using Any to avoid pyright issues with pyarrow types
    is_node: bool
    labels: Optional[list[str]] = None
    node_label: Optional[str] = (
        None  # Graph label for node files (for source-name mapping)
    )
    relationship_type: Optional[str] = None
    relationship_head: Optional[str] = None
    relationship_tail: Optional[str] = None
    # Key property info - computed once by the formatter
    key_properties: Optional[list[str]] = None  # For nodes
    head_node_key_properties: Optional[list[str]] = None  # For relationships
    tail_node_key_properties: Optional[list[str]] = None  # For relationships


@dataclass
class TypeInfo:
    """Type information for import spec generation.

    Attributes:
        source_type: Base type for source schema (e.g., 'FLOAT') - used in
            recommendedType and supportedTypes. The Java deserializer only
            accepts base types (UPPERCASE), not array types.
        target_type: Full type for target properties (e.g., 'FLOAT_ARRAY') -
            used in target_property_type. Array types are valid here (UPPERCASE).
        is_array: Whether the type is an array type.
        raw_type: The raw type as it appears in the Parquet schema (e.g., 'FLOAT').
            For arrays, this is the element type to match Parquet column metadata.
    """

    source_type: str
    target_type: str
    is_array: bool = False
    raw_type: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        """Normalize types to uppercase after initialization."""
        self.source_type = self.source_type.upper()
        self.target_type = self.target_type.upper()
        # raw_type defaults to source_type (base type) for Parquet compatibility
        self.raw_type = (self.raw_type or self.source_type).upper()


class Neo4jGraphParquetFormatter:
    """Formats Neo4j graph data into Parquet format."""

    def __init__(self, schema: Optional[dict[str, Any]] = None) -> None:
        self.schema = schema

    @staticmethod
    def _get_base_type_from_inner(inner_type: str) -> str:
        """Get the base type string from a PyArrow inner type.

        Args:
            inner_type: Inner type string from a list type (e.g., 'float64', 'string')

        Returns:
            Base type string (e.g., 'integer', 'float', 'string')
        """
        if inner_type in (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ):
            return "integer"
        if inner_type in ("float32", "float64", "double"):
            return "float"
        if inner_type in ("string", "utf8", "large_string", "large_utf8"):
            return "string"
        if inner_type == "bool":
            return "boolean"
        if inner_type in ("date32", "date64"):
            return "date"
        if inner_type.startswith("timestamp"):
            return "local_datetime"
        if inner_type in ("time32", "time64"):
            return "local_time"
        return "string"  # Default fallback

    @staticmethod
    def pyarrow_type_to_type_info(pyarrow_type: Any) -> TypeInfo:
        """Convert PyArrow type to TypeInfo with separate source and target types.

        The Java deserializer only accepts base types (e.g., 'FLOAT') for source
        schema fields (recommendedType, supportedTypes), but accepts array types
        (e.g., 'FLOAT_ARRAY') for target property types (target_property_type).

        All types are returned in UPPERCASE to match the Java enum expectations.

        Args:
            pyarrow_type: PyArrow type object or string representation

        Returns:
            TypeInfo with:
            - source_type: Base type in UPPERCASE (e.g., 'FLOAT')
            - target_type: Full type in UPPERCASE (e.g., 'FLOAT_ARRAY')
            - raw_type: Base element type in UPPERCASE for Parquet compatibility
            - is_array: Whether the type is an array type
        """
        type_str = str(pyarrow_type).lower()

        # Handle list/array types
        if type_str.startswith("list<") or type_str.startswith("large_list<"):
            inner_type = type_str.split("<", 1)[1].rstrip(">")
            if inner_type.startswith("item: "):
                inner_type = inner_type.replace("item: ", "")

            base_type = Neo4jGraphParquetFormatter._get_base_type_from_inner(inner_type)
            return TypeInfo(
                source_type=base_type,
                target_type=f"{base_type}_array",
                is_array=True,
                raw_type=base_type,  # Use base type for rawType (Parquet element type)
            )

        # Handle scalar types
        if type_str in (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ):
            return TypeInfo(
                source_type="integer", target_type="integer", is_array=False
            )

        if type_str in ("float32", "float64", "double"):
            return TypeInfo(source_type="float", target_type="float", is_array=False)

        if type_str in ("string", "utf8", "large_string", "large_utf8"):
            return TypeInfo(source_type="string", target_type="string", is_array=False)

        if type_str == "bool":
            return TypeInfo(
                source_type="boolean", target_type="boolean", is_array=False
            )

        if type_str in ("date32", "date64"):
            return TypeInfo(source_type="date", target_type="date", is_array=False)

        if type_str in (
            "timestamp",
            "timestamp[ns]",
            "timestamp[us]",
            "timestamp[ms]",
            "timestamp[s]",
        ):
            return TypeInfo(
                source_type="local_datetime",
                target_type="local_datetime",
                is_array=False,
            )

        if type_str in ("time32", "time64"):
            return TypeInfo(
                source_type="local_time", target_type="local_time", is_array=False
            )

        # Default fallback
        return TypeInfo(source_type="string", target_type="string", is_array=False)

    def _nodes_to_rows(
        self,
        nodes: list[Neo4jNode],
        lexical_graph_config: LexicalGraphConfig,
    ) -> DefaultDict[str, list[dict[str, Any]]]:
        """Convert Neo4jNode objects to row dictionaries."""
        label_to_rows: DefaultDict[str, list[dict[str, Any]]] = defaultdict(list)

        for node in nodes:
            labels: list[str] = [node.label]
            if node.label not in lexical_graph_config.lexical_graph_node_labels:
                labels.append("__Entity__")

            row: dict[str, Any] = {
                "__id__": node.id,
                "labels": labels,
            }

            if node.properties:
                row.update(node.properties)
            if node.embedding_properties:
                row.update(node.embedding_properties)

            label_to_rows[node.label].append(row)

        return label_to_rows

    def _get_key_property_name_for_label(self, node_label: str) -> Optional[str]:
        """Get the primary key property name for a node label from schema constraints.

        Args:
            node_label: The label of the node type

        Returns:
            The property name that is the primary key, or None if using default "__id__"
        """
        unique_props = get_unique_properties_for_node_type(self.schema, node_label)
        # If the only property is "__id__" (the default), return None to use node.id
        if unique_props == ["__id__"]:
            return None
        return unique_props[0]

    def _get_node_key_property_value(self, node: Neo4jNode) -> Any:
        """Get the primary key property value for a node.

        Uses schema constraints to find the primary key property name,
        then returns the value of that property from the node.

        Args:
            node: The Neo4jNode to get the key value from

        Returns:
            The value of the primary key property, or node.id if no constraint is found

        Raises:
            ValueError: If the node is missing the key property or if the property value is null
        """
        key_prop = self._get_key_property_name_for_label(node.label)
        if not key_prop:
            # there is no key property, we use the node ID
            return node.id
        if key_prop not in node.properties:
            # the Key property is missing from the node properties
            # we cannot use this node
            # Note: this should not happen as the key property is also required
            raise ValueError(f"Missing key property {key_prop} on node {node.id}")
        property_value = node.properties[key_prop]
        if property_value is None:
            # the value of the key property is null, we cannot use this node
            raise ValueError(f"Key property {key_prop} on node {node.id} is null")
        return property_value

    def _relationships_to_rows(
        self,
        relationships: list[Neo4jRelationship],
        node_id_to_node: dict[str, Neo4jNode],
    ) -> DefaultDict[tuple[str, str, str], list[dict[str, Any]]]:
        """Convert Neo4jRelationship objects to row dictionaries.

        Args:
            relationships: List of relationships to convert
            node_id_to_node: Mapping from node ID to Neo4jNode for O(1) lookups

        Returns:
            Dictionary mapping (rel_type, head_label, tail_label) tuple to list of row dictionaries.
            This ensures each file has consistent key properties for import spec generation.
        """
        type_to_rows: DefaultDict[tuple[str, str, str], list[dict[str, Any]]] = (
            defaultdict(list)
        )

        for rel in relationships:
            start_node = node_id_to_node.get(rel.start_node_id)
            if start_node is None:
                raise ValueError(
                    f"Relationship references unknown start node: {rel.start_node_id}"
                )
            end_node = node_id_to_node.get(rel.end_node_id)
            if end_node is None:
                raise ValueError(
                    f"Relationship references unknown end node: {rel.end_node_id}"
                )

            try:
                from_id = self._get_node_key_property_value(start_node)
                to_id = self._get_node_key_property_value(end_node)
            except ValueError as e:
                logger.warning(
                    "Skipping relationship (%s)-[%s]->(%s) due to bad node key property: %s",
                    rel.start_node_id,
                    rel.type,
                    rel.end_node_id,
                    e,
                )
                continue

            row: dict[str, Any] = {
                "from": from_id,
                "to": to_id,
                "from_label": start_node.label,
                "to_label": end_node.label,
                "type": rel.type,
            }

            if rel.properties:
                row.update(rel.properties)
            if rel.embedding_properties:
                row.update(rel.embedding_properties)

            # Group by (rel_type, head_label, tail_label) for consistent key properties
            key = (rel.type, start_node.label, end_node.label)
            type_to_rows[key].append(row)

        return type_to_rows

    @staticmethod
    def _normalize_column_types(rows: list[dict[str, Any]]) -> None:
        """Coerce mixed-type columns in *rows* in-place so PyArrow can build the table.

        PyArrow infers the column type from the first row; if subsequent rows have a
        different Python type for the same column the table creation fails.  This method
        detects those mismatches and coerces:
        - {int, float} -> float  (lossless numeric promotion)
        - anything else mixed -> str  (universal safe fallback)
        """
        if len(rows) <= 1:
            return

        col_types: dict[str, set[type]] = defaultdict(set)
        for row in rows:
            for key, value in row.items():
                if value is not None:
                    col_types[key].add(type(value))

        cols_to_coerce: dict[str, type] = {}
        for col, types in col_types.items():
            if len(types) <= 1:
                continue
            target: type = float if types <= {int, float} else str
            cols_to_coerce[col] = target
            logger.warning(
                "Mixed types for property '%s': %s — coercing to %s",
                col,
                {t.__name__ for t in types},
                target.__name__,
            )

        for row in rows:
            for col, target_type in cols_to_coerce.items():
                if col in row and row[col] is not None:
                    try:
                        row[col] = target_type(row[col])
                    except (ValueError, TypeError):
                        row[col] = str(row[col])

    def format_parquet(
        self,
        rows: list[dict[str, Any]],
        entity_name: str,
    ) -> tuple[bytes, Any]:
        """Format rows as Parquet bytes.

        Args:
            rows: List of row dictionaries to format
            entity_name: Name for error messages (e.g., "node label 'Person'")

        Returns:
            Tuple of (Parquet file content as bytes, PyArrow schema)

        Raises:
            ValueError: If Parquet table creation fails
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            self._normalize_column_types(rows)
            table = pa.Table.from_pylist(rows)
            # Write to BytesIO buffer
            buffer = BytesIO()
            pq.write_table(table, buffer)
            buffer.seek(0)
            return buffer.read(), table.schema
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to create Parquet table for {entity_name}: {e}"
            ) from e

    def format_graph(
        self,
        graph: Neo4jGraph,
        lexical_graph_config: LexicalGraphConfig,
        prefix: str = "",
    ) -> tuple[
        dict[str, dict[str, bytes]],
        list[FileMetadata],
        dict[str, dict[str, int]],
    ]:
        """Format the given graph as Parquet data.

        Args:
            graph: The Neo4j graph to format
            lexical_graph_config: Configuration for lexical graph processing
            prefix: Optional prefix for filenames

        Returns:
            Tuple of:
            - Dictionary with 'nodes' and 'relationships' keys, each containing
              a dictionary mapping filenames to Parquet bytes
            - List of FileMetadata objects with schema information
            - Statistics dictionary with 'nodes_per_label' and 'rel_per_type' counts
        """
        label_to_rows: DefaultDict[str, list[dict[str, Any]]] = self._nodes_to_rows(
            graph.nodes, lexical_graph_config
        )
        node_id_to_node: dict[str, Neo4jNode] = {node.id: node for node in graph.nodes}
        type_to_rows: DefaultDict[tuple[str, str, str], list[dict[str, Any]]] = (
            self._relationships_to_rows(graph.relationships, node_id_to_node)
        )

        # Format node Parquet files
        nodes_data: dict[str, bytes] = {}
        file_metadata: list[FileMetadata] = []

        for label, rows in label_to_rows.items():
            current_label: str = f"{prefix}_{label}" if prefix else label
            safe_stem = sanitize_parquet_filestem(current_label)
            filename = f"{safe_stem}.parquet"
            parquet_bytes, schema = self.format_parquet(
                rows, f"node label '{current_label}'"
            )
            nodes_data[filename] = parquet_bytes

            # Store metadata
            labels_list = [current_label]
            if current_label not in lexical_graph_config.lexical_graph_node_labels:
                labels_list.append("__Entity__")

            file_metadata.append(
                FileMetadata(
                    filename=filename,
                    schema=schema,
                    is_node=True,
                    labels=labels_list,
                    node_label=label,
                    key_properties=get_unique_properties_for_node_type(
                        self.schema, label
                    ),
                )
            )

        # Format relationship Parquet files
        # Key is (rel_type, head_label, tail_label) for consistent key properties per file
        relationships_data: dict[str, bytes] = {}
        for (rtype, head_label, tail_label), rows in type_to_rows.items():
            # Filename pattern: {head_label}_{rel_type}_{tail_label}.parquet (sanitized)
            base_name = f"{head_label}_{rtype}_{tail_label}"
            current_name: str = f"{prefix}_{base_name}" if prefix else base_name
            safe_stem = sanitize_parquet_filestem(current_name)
            filename = f"{safe_stem}.parquet"
            parquet_bytes, schema = self.format_parquet(
                rows, f"relationship '{current_name}'"
            )
            relationships_data[filename] = parquet_bytes
            # Store metadata - head/tail labels come from the key, guaranteed consistent
            file_metadata.append(
                FileMetadata(
                    filename=filename,
                    schema=schema,
                    is_node=False,
                    relationship_type=rtype,
                    relationship_head=head_label,
                    relationship_tail=tail_label,
                    head_node_key_properties=get_unique_properties_for_node_type(
                        self.schema, head_label
                    ),
                    tail_node_key_properties=get_unique_properties_for_node_type(
                        self.schema, tail_label
                    ),
                )
            )

        # Stats: rel_per_type keyed by relationship type only (aggregate counts)
        rel_per_type: dict[str, int] = {}
        for (rtype, _head_label, _tail_label), rows in type_to_rows.items():
            rel_per_type[rtype] = rel_per_type.get(rtype, 0) + len(rows)
        stats: dict[str, dict[str, int]] = {
            "nodes_per_label": {
                label: len(rows) for label, rows in label_to_rows.items()
            },
            "rel_per_type": rel_per_type,
        }

        return (
            {
                "nodes": nodes_data,
                "relationships": relationships_data,
            },
            file_metadata,
            stats,
        )
