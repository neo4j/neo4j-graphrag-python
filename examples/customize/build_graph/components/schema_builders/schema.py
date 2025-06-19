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
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
)
from neo4j_graphrag.experimental.components.types import (
    NodeType,
    PropertyType,
    RelationshipType,
)


async def main() -> None:
    schema_builder = SchemaBuilder()

    result = await schema_builder.run(
        node_types=[
            NodeType(
                label="Person",
                properties=[
                    PropertyType(name="name", type="STRING"),
                    PropertyType(name="place_of_birth", type="STRING"),
                    PropertyType(name="date_of_birth", type="DATE"),
                ],
            ),
            NodeType(
                label="Organization",
                properties=[
                    PropertyType(name="name", type="STRING"),
                    PropertyType(name="country", type="STRING"),
                ],
            ),
        ],
        relationship_types=[
            RelationshipType(
                label="WORKED_ON",
            ),
            RelationshipType(
                label="WORKED_FOR",
            ),
        ],
        patterns=[
            ("Person", "WORKED_ON", "Field"),
            ("Person", "WORKED_FOR", "Organization"),
        ],
    )
    print(result)
