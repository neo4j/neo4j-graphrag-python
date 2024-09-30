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
import abc
from typing import Optional, Union

import neo4j

from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    CHUNK_NODE_LABEL,
    DOCUMENT_NODE_LABEL,
    NODE_TO_CHUNK_RELATIONSHIP_TYPE,
)
from neo4j_graphrag.experimental.components.types import ResolutionStats
from neo4j_graphrag.experimental.pipeline import Component
from neo4j_graphrag.utils import execute_query


class EntityResolver(Component, abc.ABC):
    def __init__(self, driver: Union[neo4j.Driver, neo4j.AsyncDriver]) -> None:
        self.driver = driver

    @abc.abstractmethod
    async def run(self, document_path: str) -> ResolutionStats:
        pass


class SinglePropertyExactMatchResolver(EntityResolver):
    """Resolve entities with same labels"""

    DEFAULT_MATCH_QUERY = (
        f"MATCH (n:__Entity__)"
        f"-[:{NODE_TO_CHUNK_RELATIONSHIP_TYPE}]->(:{CHUNK_NODE_LABEL})"
        f"-->(:{DOCUMENT_NODE_LABEL} {{path: $path}})"
    )

    def __init__(
        self,
        driver: Union[neo4j.Driver, neo4j.AsyncDriver],
        resolve_property: str = "name",
        nodes_to_resolve_query: str = DEFAULT_MATCH_QUERY,
        database: Optional[str] = None,
    ) -> None:
        super().__init__(driver)
        self.resolve_property = resolve_property
        self.nodes_to_resolve_query = nodes_to_resolve_query
        self.database = database

    async def run(
        self,
        document_path: str,
    ) -> ResolutionStats:
        stat_query = f"{self.nodes_to_resolve_query} RETURN count(n) as c"
        records = await execute_query(
            self.driver,
            stat_query,
            parameters_={"path": document_path},
            database_=self.database,
        )
        number_of_affected_nodes = records[0].get("c")
        if number_of_affected_nodes == 0:
            return ResolutionStats(
                number_of_affected_nodes=0,
                number_of_created_nodes=0,
            )
        merge_nodes_query = (
            f"{self.nodes_to_resolve_query}"
            f"WITH n, n.{self.resolve_property} as prop "
            "WITH n, prop WHERE prop IS NOT NULL "
            "UNWIND labels(n) as lab  "
            "WITH lab, prop, n WHERE lab <> '__Entity__' "
            "WITH prop, lab, collect(n) AS nodes "
            "CALL apoc.refactor.mergeNodes(nodes,{ "
            " properties:'discard', "
            " mergeRels:true "
            "}) "
            "YIELD node "
            "RETURN count(node) as c "
        )
        records = await execute_query(
            self.driver,
            merge_nodes_query,
            parameters_={"path": document_path},
            database_=self.database,
        )
        number_of_created_nodes = records[0].get("c")
        return ResolutionStats(
            number_of_affected_nodes=number_of_affected_nodes,
            number_of_created_nodes=number_of_created_nodes,
        )
