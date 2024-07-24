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
from .components import ComponentMultiply


def test_component_inputs() -> None:
    inputs = ComponentMultiply.component_inputs  # type: ignore
    assert "number1" in inputs
    assert inputs["number1"]["has_default"] is False
    assert "number2" in inputs
    assert inputs["number2"]["has_default"] is True


def test_component_outputs() -> None:
    outputs = ComponentMultiply.component_outputs  # type: ignore
    assert "result" in outputs
