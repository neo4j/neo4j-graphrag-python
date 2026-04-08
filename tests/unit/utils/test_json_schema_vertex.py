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

from pydantic import BaseModel, ConfigDict, Field

from neo4j_graphrag.utils.json_schema_vertex import (
    sanitize_json_schema_for_vertex,
    strip_json_schema_null_anyof_for_vertex,
    strip_json_schema_deprecated_for_vertex,
)


class _ModelWithOptional(BaseModel):
    model_config = ConfigDict(extra="forbid")
    required_field: str
    optional_field: str | None = None


def test_strip_json_schema_null_anyof_removes_null_branch_from_optional() -> None:
    schema = _ModelWithOptional.model_json_schema()
    dumped_before = json.dumps(schema)
    assert '"type": "null"' in dumped_before

    strip_json_schema_null_anyof_for_vertex(schema)
    dumped_after = json.dumps(schema)
    assert '"type": "null"' not in dumped_after
    opt = schema["properties"]["optional_field"]
    assert opt.get("type") == "string"


class _ModelWithDeprecatedField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: str
    old: str = Field(deprecated=True)


def test_strip_json_schema_deprecated_removes_deprecated_keyword() -> None:
    schema = _ModelWithDeprecatedField.model_json_schema()
    assert '"deprecated"' in json.dumps(schema)

    strip_json_schema_deprecated_for_vertex(schema)
    assert '"deprecated"' not in json.dumps(schema)
    assert schema["properties"]["old"].get("type") == "string"


class _ModelOptionalAndDeprecated(BaseModel):
    model_config = ConfigDict(extra="forbid")
    required: str
    optional: str | None = None
    legacy: str = Field(deprecated=True)


def test_sanitize_json_schema_for_vertex_applies_null_and_deprecated_strips() -> None:
    schema = _ModelOptionalAndDeprecated.model_json_schema()
    dumped_before = json.dumps(schema)
    assert '"type": "null"' in dumped_before
    assert '"deprecated"' in dumped_before

    sanitize_json_schema_for_vertex(schema)
    dumped_after = json.dumps(schema)
    assert '"type": "null"' not in dumped_after
    assert '"deprecated"' not in dumped_after
    assert schema["properties"]["optional"].get("type") == "string"
    assert schema["properties"]["legacy"].get("deprecated") is None
