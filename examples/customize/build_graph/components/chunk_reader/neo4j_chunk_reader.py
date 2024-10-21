import asyncio

import neo4j
from neo4j_graphrag.experimental.components.neo4j_reader import Neo4jChunkReader
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig, TextChunks


async def main(driver: neo4j.Driver) -> TextChunks:
    config = LexicalGraphConfig(
        chunk_node_label="TextPart",
    )
    reader = Neo4jChunkReader(driver)
    result = await reader.run(lexical_graph_config=config)
    return result


if __name__ == "__main__":
    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    ) as driver:
        print(asyncio.run(main(driver)))
