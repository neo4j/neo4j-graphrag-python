import neo4j
from neo4j_graphrag.experimental.components.kg_writer import (
    KGWriterModel,
    Neo4jWriter,
)
from neo4j_graphrag.experimental.components.types import Neo4jGraph


async def main(driver: neo4j.Driver, graph: Neo4jGraph) -> KGWriterModel:
    writer = Neo4jWriter(
        driver,
        # optionally, configure the neo4j database
        # neo4j_database="neo4j",
        # you can tune batch_size to
        # improve speed
        # batch_size=1000,
    )
    result = await writer.run(graph=graph)
    return result
