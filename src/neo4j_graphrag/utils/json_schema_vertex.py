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

"""JSON Schema tweaks for Vertex AI ``response_schema`` / ``GenerationConfig``."""

from __future__ import annotations

from typing import Any


def strip_json_schema_null_anyof_for_vertex(obj: Any) -> None:
    """Remove ``{\"type\": \"null\"}`` branches from ``anyOf`` (recursive, in place).

    Pydantic emits ``anyOf: [<T>, {\"type\": \"null\"}]`` for ``Optional`` fields.
    Vertex AI's ``response_schema`` protobuf parser rejects ``type: NULL``, so we
    collapse optional scalars to the non-null branch only. Validation still accepts
    omitted keys or empty strings at runtime.

    Used by :class:`~neo4j_graphrag.experimental.components.schema.GraphSchema`
    and by :class:`~neo4j_graphrag.llm.vertexai_llm.VertexAILLM` so structured
    output works even when callers pass plain ``BaseModel`` types or raw schemas.
    """
    if isinstance(obj, dict):
        if "anyOf" in obj and isinstance(obj["anyOf"], list):
            branches = obj["anyOf"]
            non_null = [
                b
                for b in branches
                if not (isinstance(b, dict) and b.get("type") == "null")
            ]
            if len(non_null) == 1 and isinstance(non_null[0], dict):
                branch = non_null[0]
                preserved = {k: v for k, v in obj.items() if k != "anyOf"}
                preserved.pop("default", None)
                preserved.update(branch)
                obj.clear()
                obj.update(preserved)
            elif len(non_null) > 1:
                obj["anyOf"] = non_null
        if obj.get("default") is None and "default" in obj:
            del obj["default"]
        for v in obj.values():
            strip_json_schema_null_anyof_for_vertex(v)
    elif isinstance(obj, list):
        for item in obj:
            strip_json_schema_null_anyof_for_vertex(item)


def strip_json_schema_deprecated_for_vertex(obj: Any) -> None:
    """Remove ``deprecated`` keys from schema objects (recursive, in place).

    Pydantic v2 adds ``deprecated: true`` to JSON Schema for fields declared with
    ``Field(deprecated=...)``. Vertex AI's ``Schema`` protobuf has no ``deprecated``
    field and :func:`google.protobuf.json_format.ParseDict` rejects it.
    """
    if isinstance(obj, dict):
        obj.pop("deprecated", None)
        for v in obj.values():
            strip_json_schema_deprecated_for_vertex(v)
    elif isinstance(obj, list):
        for item in obj:
            strip_json_schema_deprecated_for_vertex(item)


def sanitize_json_schema_for_vertex(obj: Any) -> None:
    """Normalize JSON Schema for Vertex ``response_schema`` / ``GenerationConfig``.

    Applies :func:`strip_json_schema_null_anyof_for_vertex` and
    :func:`strip_json_schema_deprecated_for_vertex`. Call this (or the individual
    strippers) after building a schema from Pydantic or user dicts.
    """
    strip_json_schema_null_anyof_for_vertex(obj)
    strip_json_schema_deprecated_for_vertex(obj)
