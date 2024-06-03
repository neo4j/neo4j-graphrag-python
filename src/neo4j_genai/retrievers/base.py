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
from abc import ABC, abstractmethod
from typing import Optional, Any
import neo4j

from neo4j_genai.exceptions import Neo4jVersionError


class Retriever(ABC):
    """
    Abstract class for Neo4j retrievers
    """

    index_name: str

    def __init__(self, driver: neo4j.Driver):
        self.driver = driver
        self._verify_version()

    def _verify_version(self) -> None:
        """
        Check if the connected Neo4j database version supports vector indexing.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.18.1) that is known to support vector
        indexing. Raises a Neo4jMinVersionError if the connected Neo4j version is
        not supported.
        """
        records, _, _ = self.driver.execute_query("CALL dbms.components()")
        version = records[0]["versions"][0]

        if "aura" in version:
            version_tuple = (
                *tuple(map(int, version.split("-")[0].split("."))),
                0,
            )
            target_version = (5, 18, 0)
        else:
            version_tuple = tuple(map(int, version.split(".")))
            target_version = (5, 18, 1)

        if version_tuple < target_version:
            raise Neo4jVersionError()

    @abstractmethod
    def search(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def _fetch_index_infos(self) -> None:
        """Fetch the node label and embedding property from the index definition"""
        query = (
            "SHOW VECTOR INDEXES "
            "YIELD name, labelsOrTypes, properties, options "
            "WHERE name = $index_name "
            "RETURN labelsOrTypes as labels, properties, "
            "options.indexConfig.`vector.dimensions` as dimensions"
        )
        query_result = self.driver.execute_query(query, {"index_name": self.index_name})
        try:
            result = query_result.records[0]
            self._node_label = result["labels"][0]
            self._embedding_node_property = result["properties"][0]
            self._embedding_dimension = result["dimensions"]
        except IndexError:
            raise Exception(f"No index with name {self.index_name} found")


class ExternalRetriever(ABC):
    """
    Abstract class for External Vector Stores
    """

    def __init__(self, id_property_external: str, id_property_neo4j: str) -> None:
        self.id_property_external = id_property_external
        self.id_property_neo4j = id_property_neo4j

    @abstractmethod
    def search(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[neo4j.Record]:
        """

        Returns:
                list[neo4j.Record]: List of Neo4j Records

        """
        pass
