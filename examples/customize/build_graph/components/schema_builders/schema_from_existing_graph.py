"""This example demonstrates how to use the SchemaFromExistingGraphExtractor component
to automatically extract a schema from an existing Neo4j database.
"""

import asyncio
from pprint import pprint

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
        extractor = SchemaFromExistingGraphExtractor(
            driver,
            # optional:
            neo4j_database=DATABASE,
            additional_patterns=True,
            additional_node_types=True,
            additional_relationship_types=True,
            additional_properties=True,
        )
        schema: GraphSchema = await extractor.run()
        # schema.store_as_json("my_schema.json")
        pprint(schema.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
