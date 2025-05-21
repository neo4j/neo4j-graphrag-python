"""This example demonstrates how to use the SchemaFromExistingGraphExtractor component
to automatically extract a schema from an existing Neo4j database.
"""

import asyncio

import neo4j

from neo4j_graphrag.experimental.components.schema import (
    SchemaFromExistingGraphExtractor,
    GraphSchema,
)


URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX = "moviePlotsEmbedding"


async def main() -> None:
    """Run the example."""

    with neo4j.GraphDatabase.driver(
        URI,
        auth=AUTH,
    ) as driver:
        extractor = SchemaFromExistingGraphExtractor(driver)
        schema: GraphSchema = await extractor.run()
        # schema.store_as_json("my_schema.json")
        print(schema)


if __name__ == "__main__":
    asyncio.run(main())
