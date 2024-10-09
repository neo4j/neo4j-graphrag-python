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
from typing import Any, Optional, Union

import neo4j

from neo4j_graphrag.experimental.components.types import ResolutionStats
from neo4j_graphrag.experimental.pipeline import Component
from neo4j_graphrag.utils import execute_query


class EntityResolver(Component, abc.ABC):
    """Entity resolution base class

    Args:
        driver (neo4j.driver): The Neo4j driver to connect to the database.
        filter_query (Optional[str]): Cypher query to select the entities to resolve. By default, all nodes with __Entity__ label are used
    """

    def __init__(
        self,
        driver: Union[neo4j.Driver, neo4j.AsyncDriver],
        filter_query: Optional[str] = None,
    ) -> None:
        self.driver = driver
        self.filter_query = filter_query

    @abc.abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> ResolutionStats:
        pass


class SinglePropertyExactMatchResolver(EntityResolver):
    """Resolve entities with same label and exact same property (default is "name").

    Args:
        driver (neo4j.driver): The Neo4j driver to connect to the database.
        filter_query (Optional[str]): To reduce the resolution scope, add a Cypher WHERE clause.
        resolve_property (str): The property that will be compared (default: "name"). If values match exactly, entities are merged.
        neo4j_database (Optional[str]): The name of the Neo4j database to write to. Defaults to 'neo4j' if not provided.

    Example:

    .. code-block:: python

        from neo4j import AsyncGraphDatabase
        from neo4j_graphrag.experimental.components.resolver import SinglePropertyExactMatchResolver

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")
        DATABASE = "neo4j"

        driver = AsyncGraphDatabase.driver(URI, auth=AUTH, database=DATABASE)
        resolver = SinglePropertyExactMatchResolver(driver=driver, neo4j_database=DATABASE)
        await resolver.run()  # no expected parameters

    """

    def __init__(
        self,
        driver: Union[neo4j.Driver, neo4j.AsyncDriver],
        filter_query: Optional[str] = None,
        resolve_property: str = "name",
        neo4j_database: Optional[str] = None,
    ) -> None:
        super().__init__(driver, filter_query)
        self.resolve_property = resolve_property
        self.database = neo4j_database

    async def run(self) -> ResolutionStats:
        """Resolve entities based on the following rule:
        For each entity label, entities with the same 'resolve_property' value
        (exact match) are grouped into a single node:

        - Properties: the property from the first node will remain if already set, otherwise the first property in list will be written.
        - Relationships: merge relationships with same type and target node.

        See apoc.refactor.mergeNodes documentation for more details.
        """
        match_query = "MATCH (entity:__Entity__) "
        if self.filter_query:
            match_query += self.filter_query
        stat_query = f"{match_query} RETURN count(entity) as c"
        records = await execute_query(
            self.driver,
            stat_query,
            database_=self.database,
        )
        number_of_nodes_to_resolve = records[0].get("c")
        if number_of_nodes_to_resolve == 0:
            return ResolutionStats(
                number_of_nodes_to_resolve=0,
            )
        merge_nodes_query = (
            f"{match_query} "
            f"WITH entity, entity.{self.resolve_property} as prop "
            # keep only entities for which the resolve_property (name) is not null
            "WITH entity, prop WHERE prop IS NOT NULL "
            # will check the property for each of the entity labels,
            # except the reserved ones __Entity__ and __KGBuilder__
            "UNWIND labels(entity) as lab  "
            "WITH lab, prop, entity WHERE NOT lab IN ['__Entity__', '__KGBuilder__'] "
            # aggregate based on property value and label
            # collect all entities with exact same property and label
            # in the 'entities' list
            "WITH prop, lab, collect(entity) AS entities "
            # merge all entities into a single node
            # * merge relationships: if the merged entities have a relationship of same
            # type to the same target node, these relationships are merged
            # otherwise relationships are just attached to the newly created node
            # * properties: if the two entities have the same property key with
            # different values, only one of them is kept in the created node
            "CALL apoc.refactor.mergeNodes(entities,{ "
            " properties:'discard', "
            " mergeRels:true "
            "}) "
            "YIELD node "
            "RETURN count(node) as c "
        )
        records = await execute_query(
            self.driver,
            merge_nodes_query,
            database_=self.database,
        )
        number_of_created_nodes = records[0].get("c")
        return ResolutionStats(
            number_of_nodes_to_resolve=number_of_nodes_to_resolve,
            number_of_created_nodes=number_of_created_nodes,
        )
