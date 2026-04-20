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
from neo4j_graphrag.utils.validation import issubclass_safe


def test_issubclass_safe_direct_subclass() -> None:
    assert issubclass_safe(bool, int) is True


def test_issubclass_safe_not_subclass_returns_false() -> None:
    # Covers the `return False` branch (line 45)
    assert issubclass_safe(str, int) is False


def test_issubclass_safe_with_tuple() -> None:
    # Covers the `isinstance(class_or_tuple, tuple)` branch (line 32)
    assert issubclass_safe(bool, (str, int)) is True
    assert issubclass_safe(str, (int, float)) is False
