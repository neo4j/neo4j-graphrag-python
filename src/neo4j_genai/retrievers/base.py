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
from typing import Optional, Callable, Any
import neo4j

from neo4j_genai.types import RawSearchResult, RetrieverResult, RetrieverResultItem
from neo4j_genai.exceptions import Neo4jVersionError


class Retriever(ABC):
    """
    Abstract class for Neo4j retrievers
    """

    index_name: str
    VERIFY_NEO4J_VERSION = True

    def __init__(self, driver: neo4j.Driver):
        self.driver = driver
        if self.VERIFY_NEO4J_VERSION:
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

    def search(self, *args: Any, **kwargs: Any) -> RetrieverResult:
        """
        Search method. Call the get_search_result method that returns
        a list of neo4j.Record, and format them to return RetrieverResult.
        """
        raw_result = self._get_search_results(*args, **kwargs)
        formatter = self.get_result_formatter()
        search_items = [formatter(record) for record in raw_result.records]
        metadata = raw_result.metadata or {}
        metadata["__retriever"] = self.__class__.__name__
        return RetrieverResult(
            items=search_items,
            metadata=metadata,
        )

    @abstractmethod
    def _get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        pass

    def get_result_formatter(self) -> Callable[[neo4j.Record], RetrieverResultItem]:
        """
        Returns the function to use to transform a neo4j.Record to a RetrieverResultItem.
        """
        if hasattr(self, "format_record_function"):
            return self.format_record_function or self.default_format_record
        return self.default_format_record

    def default_format_record(self, record: neo4j.Record) -> RetrieverResultItem:
        """
        Best effort to guess the node to text method. Inherited classes
        can override this method to implement custom text formatting.
        """
        return RetrieverResultItem(content=str(record), metadata=record.get("metadata"))


class ExternalRetriever(Retriever, ABC):
    """
    Abstract class for External Vector Stores
    """

    VERIFY_NEO4J_VERSION = False

    def __init__(
        self, driver: neo4j.Driver, id_property_external: str, id_property_neo4j: str
    ):
        super().__init__(driver)
        self.id_property_external = id_property_external
        self.id_property_neo4j = id_property_neo4j

    @abstractmethod
    def _get_search_results(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RawSearchResult:
        """

        Returns:
                list[neo4j.Record]: List of Neo4j Records

        """
        pass
