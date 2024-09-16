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
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
)

schema_builder = SchemaBuilder()

schema_builder.run(
    entities=[
        SchemaEntity(
            label="Person",
            properties=[
                SchemaProperty(name="name", type="STRING"),
                SchemaProperty(name="place_of_birth", type="STRING"),
                SchemaProperty(name="date_of_birth", type="DATE"),
            ],
        ),
        SchemaEntity(
            label="Organization",
            properties=[
                SchemaProperty(name="name", type="STRING"),
                SchemaProperty(name="country", type="STRING"),
            ],
        ),
    ],
    relations=[
        SchemaRelation(
            label="WORKED_ON",
        ),
        SchemaRelation(
            label="WORKED_FOR",
        ),
    ],
    potential_schema=[
        ("Person", "WORKED_ON", "Field"),
        ("Person", "WORKED_FOR", "Organization"),
    ],
)
