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
import asyncio

import neo4j

from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
)


async def main() -> None:
    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password"),
    ) as driver:
        schema_builder = SchemaBuilder(driver)

        schema = await schema_builder.run(
            node_types=[
                {
                    "label": "Person",
                    "properties": [
                        {"name": "name", "type": "STRING"},
                        {"name": "place_of_birth", "type": "STRING"},
                        {"name": "date_of_birth", "type": "DATE"},
                    ],
                },
                {
                    "label": "Organization",
                    "properties": [
                        {"name": "name", "type": "STRING"},
                        {"name": "country", "type": "STRING"},
                    ],
                },
                {
                    "label": "Field",
                    "properties": [
                        {"name": "name", "type": "STRING"},
                    ],
                },
            ],
            relationship_types=[
                "WORKED_ON",
                {
                    "label": "WORKED_FOR",
                },
            ],
            patterns=[
                ("Person", "WORKED_ON", "Field"),
                ("Person", "WORKED_FOR", "Organization"),
            ],
        )
        print(schema)


if __name__ == "__main__":
    asyncio.run(main())
