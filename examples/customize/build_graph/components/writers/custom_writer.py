"""Implement a custom writer to save the results, for instance by using
custom Cypher queries.
"""

import neo4j
from neo4j_graphrag.experimental.components.kg_writer import KGWriter, KGWriterModel
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig, Neo4jGraph
from pydantic import validate_call
from neo4j_graphrag.utils import driver_config


class MyWriter(KGWriter):
    def __init__(self, driver: neo4j.Driver) -> None:
        self.driver = driver_config.override_user_agent(driver)

    @validate_call
    async def run(
        self,
        graph: Neo4jGraph,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> KGWriterModel:
        try:
            self.driver.execute_query("my query")
            return KGWriterModel(status="SUCCESS")
        except Exception:
            return KGWriterModel(status="FAILURE")
