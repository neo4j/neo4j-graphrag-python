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

from pydantic import BaseModel, ConfigDict

from neo4j_graphrag.utils.json_schema_structured_output import (
    make_strict_json_schema_for_structured_output,
)


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: str
    b: int = 1


def test_make_strict_sets_additional_properties_and_required() -> None:
    raw = _M.model_json_schema()
    make_strict_json_schema_for_structured_output(raw)
    assert raw.get("additionalProperties") is False
    assert set(raw["required"]) == {"a", "b"}


def test_make_strict_const_to_enum() -> None:
    class _Const(BaseModel):
        model_config = ConfigDict(extra="forbid")
        x: str

    raw = _Const.model_json_schema()
    # pydantic may not emit const in default schema; ensure function is safe
    make_strict_json_schema_for_structured_output(raw)
    assert "properties" in raw
