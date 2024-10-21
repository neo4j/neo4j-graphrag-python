"""Implement a custom writer to save the results, for instance by using
custom Cypher queries.
"""

import neo4j
from neo4j_graphrag.experimental.components.kg_writer import KGWriter, KGWriterModel
from neo4j_graphrag.experimental.components.types import Neo4jGraph
from pydantic import validate_call


class MyWriter(KGWriter):
    def __init__(self, driver: neo4j.Driver) -> None:
        self.driver = driver

    @validate_call
    async def run(self, graph: Neo4jGraph) -> KGWriterModel:
        try:
            self.driver.execute_query("my query")
            return KGWriterModel(status="SUCCESS")
        except Exception:
            return KGWriterModel(status="FAILURE")
