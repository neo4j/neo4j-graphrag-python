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
import warnings
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


def _constraint_relationship_type_unset(constraint: dict[str, Any]) -> bool:
    rt = constraint.get("relationship_type")
    return rt is None or (isinstance(rt, str) and rt.strip() == "")


def _resolve_constraint_property_names(constraint: dict[str, Any]) -> list[str]:
    """Resolve property names from a constraint dict (``property_names`` or ``property_name``)."""
    pns = constraint.get("property_names") or ()
    if pns:
        return list(pns)
    pn = constraint.get("property_name", "")
    return [pn] if pn else []


def get_uniqueness_property_names_for_node_type(
    schema: Optional[dict[str, Any]], node_label: str
) -> list[str]:
    """Property names with a UNIQUENESS constraint for this node label (flat, order as in schema)."""
    if not schema:
        return []
    out: list[str] = []
    for constraint in schema.get("constraints", ()) or ():
        if constraint.get("type") != "UNIQUENESS":
            continue
        if constraint.get("node_type", "") != node_label:
            continue
        out.extend(_resolve_constraint_property_names(constraint))
    return out


def get_key_property_names_for_node_type(
    schema: Optional[dict[str, Any]], node_label: str
) -> list[str]:
    """Property names with a KEY constraint (node scope) for this node label (flat)."""
    if not schema:
        return []
    out: list[str] = []
    for constraint in schema.get("constraints", ()) or ():
        if constraint.get("type") != "KEY":
            continue
        if constraint.get("node_type", "") != node_label:
            continue
        if not _constraint_relationship_type_unset(constraint):
            continue
        out.extend(_resolve_constraint_property_names(constraint))
    return out


def get_key_constraints_for_node_type(
    schema: Optional[dict[str, Any]], node_label: str
) -> list[tuple[str, ...]]:
    """KEY constraints for a node label, preserving composite grouping.

    Returns a list of tuples, each containing the property names for one KEY constraint.
    """
    if not schema:
        return []
    out: list[tuple[str, ...]] = []
    for constraint in schema.get("constraints", ()) or ():
        if constraint.get("type") != "KEY":
            continue
        if constraint.get("node_type", "") != node_label:
            continue
        if not _constraint_relationship_type_unset(constraint):
            continue
        props = _resolve_constraint_property_names(constraint)
        if props:
            out.append(tuple(props))
    return out


def get_uniqueness_constraints_for_node_type(
    schema: Optional[dict[str, Any]], node_label: str
) -> list[tuple[str, ...]]:
    """UNIQUENESS constraints for a node label, preserving composite grouping.

    Returns a list of tuples, each containing the property names for one UNIQUENESS constraint.
    """
    if not schema:
        return []
    out: list[tuple[str, ...]] = []
    for constraint in schema.get("constraints", ()) or ():
        if constraint.get("type") != "UNIQUENESS":
            continue
        if constraint.get("node_type", "") != node_label:
            continue
        props = _resolve_constraint_property_names(constraint)
        if props:
            out.append(tuple(props))
    return out


def get_primary_key_column_names_for_node_type(
    schema: Optional[dict[str, Any]], node_label: str
) -> list[str]:
    """Column names flagged as primary key in KG writer metadata: KEY properties, else ``__id__``."""
    keys = get_key_property_names_for_node_type(schema, node_label)
    if keys:
        return keys
    return ["__id__"]


def get_unique_properties_for_node_type(
    schema: Optional[dict[str, Any]], node_label: str
) -> list[str]:
    """Deprecated synonym for :func:`get_primary_key_column_names_for_node_type`.

    Historically this returned UNIQUENESS-backed property names (with a ``__id__``
    fallback). It now follows **primary-key** semantics (KEY constraints, else
    ``__id__``). Use :func:`get_uniqueness_property_names_for_node_type` or
    :func:`get_primary_key_column_names_for_node_type` instead.
    """
    warnings.warn(
        "get_unique_properties_for_node_type is deprecated and its meaning has "
        "changed: it now mirrors get_primary_key_column_names_for_node_type (KEY / "
        "__id__), not UNIQUENESS-only lists. Use "
        "get_uniqueness_property_names_for_node_type or "
        "get_primary_key_column_names_for_node_type.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_primary_key_column_names_for_node_type(schema, node_label)


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
    # Schema-driven column roles for KGWriter metadata (see ParquetWriter)
    primary_key_property_names: Optional[list[str]] = None
    uniqueness_property_names: Optional[list[str]] = None
    head_primary_key_property_names: Optional[list[str]] = None
    head_uniqueness_property_names: Optional[list[str]] = None
    tail_primary_key_property_names: Optional[list[str]] = None
    tail_uniqueness_property_names: Optional[list[str]] = None
    # Grouped constraint metadata (preserves composite grouping)
    key_constraints: Optional[list[tuple[str, ...]]] = None
    uniqueness_constraints: Optional[list[tuple[str, ...]]] = None


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

    def _get_identity_property_name_for_label(self, node_label: str) -> Optional[str]:
        """Resolve natural identity: first KEY property, else first UNIQUENESS, else None (use ``__id__``)."""
        key_props = get_key_property_names_for_node_type(self.schema, node_label)
        if key_props:
            return key_props[0]
        uq = get_uniqueness_property_names_for_node_type(self.schema, node_label)
        if uq:
            return uq[0]
        return None

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
        key_prop = self._get_identity_property_name_for_label(node.label)
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

            # Build an explicit schema from the union of all keys to avoid
            # silent column drops when the first row lacks some keys (e.g. embeddings).
            # Dict preserves first-seen insertion order for deterministic column ordering.
            all_keys: dict[str, None] = {k: None for row in rows for k in row}
            # Collect the first non-null, non-empty-list value per key for type inference.
            # Also track keys that carry at least one empty list, so we can fall back to
            # list<null> when no richer sample exists (rather than pa.null(), which cannot
            # hold [] values).
            sample: dict[str, Any] = {}
            has_empty_list: set[str] = set()
            for row in rows:
                for k, v in row.items():
                    if v == []:
                        has_empty_list.add(k)
                    elif k not in sample and v is not None:
                        sample[k] = v

            fields: list[pa.Field] = []
            for k in all_keys:
                if k in sample:
                    t: Any = pa.infer_type([sample[k]])
                    if pa.types.is_list(t) and (
                        pa.types.is_floating(t.value_type)
                        or pa.types.is_integer(t.value_type)
                    ):
                        t = pa.list_(pa.float32())
                elif k in has_empty_list:
                    # Only empty lists seen — use list<null> so [] values round-trip correctly.
                    # Note: list<null> is a known limitation; some consumers (DuckDB, Spark)
                    # may not handle this type well.
                    t = pa.list_(pa.null())
                else:
                    t = pa.null()
                fields.append(pa.field(k, t))

            schema = pa.schema(fields) if fields else None
            table = pa.Table.from_pylist(rows, schema=schema)

            buffer = BytesIO()
            pq.write_table(table, buffer)
            buffer.seek(0)
            return buffer.read(), table.schema
        except ImportError:
            raise
        except (ValueError, TypeError, pa.ArrowInvalid) as e:
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
                    primary_key_property_names=get_key_property_names_for_node_type(
                        self.schema, label
                    ),
                    uniqueness_property_names=get_uniqueness_property_names_for_node_type(
                        self.schema, label
                    ),
                    key_constraints=get_key_constraints_for_node_type(
                        self.schema, label
                    ),
                    uniqueness_constraints=get_uniqueness_constraints_for_node_type(
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
                    head_primary_key_property_names=get_key_property_names_for_node_type(
                        self.schema, head_label
                    ),
                    head_uniqueness_property_names=get_uniqueness_property_names_for_node_type(
                        self.schema, head_label
                    ),
                    tail_primary_key_property_names=get_key_property_names_for_node_type(
                        self.schema, tail_label
                    ),
                    tail_uniqueness_property_names=get_uniqueness_property_names_for_node_type(
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
