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

"""JSON Schema post-processing for LLM structured output (OpenAI / Vertex AI)."""

from __future__ import annotations

from typing import Any


def make_strict_json_schema_for_structured_output(schema: dict[str, Any]) -> None:
    """Mutate *schema* in place for provider compatibility.

    OpenAI requires ``additionalProperties: false`` and every key in ``properties``
    listed in ``required``. Vertex AI rejects ``const`` in favor of single-value
    ``enum``. Applied recursively, including ``$defs``.
    """

    def make_strict(obj: dict[str, Any]) -> None:
        if obj.get("type") == "object" and "properties" in obj:
            obj["additionalProperties"] = False
            obj["required"] = list(obj["properties"].keys())

        if "const" in obj:
            obj["enum"] = [obj.pop("const")]

        for value in obj.values():
            if isinstance(value, dict):
                make_strict(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        make_strict(item)

    make_strict(schema)
    defs = schema.get("$defs")
    if isinstance(defs, dict):
        for def_schema in defs.values():
            if isinstance(def_schema, dict):
                make_strict(def_schema)
