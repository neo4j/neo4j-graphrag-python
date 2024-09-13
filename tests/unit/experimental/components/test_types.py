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
import pytest
from neo4j_graphrag.experimental.components.types import Neo4jNode


def test_neo4j_node_invalid_property() -> None:
    with pytest.raises(TypeError) as excinfo:
        Neo4jNode(id="0", label="Label", properties={"id": "1"})
        assert "'id' as a property name is not allowed" in str(excinfo)


def test_neo4j_node_invalid_embedding_property() -> None:
    with pytest.raises(TypeError) as excinfo:
        Neo4jNode(id="0", label="Label", embedding_properties={"id": [1.0, 2.0, 3.0]})
        assert "'id' as a property name is not allowed" in str(excinfo)
