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
from __future__ import annotations

import inspect
import types
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, Optional, TypeVar

import neo4j
from typing_extensions import ParamSpec

from neo4j_graphrag.exceptions import Neo4jVersionError
from neo4j_graphrag.types import RawSearchResult, RetrieverResult, RetrieverResultItem

T = ParamSpec("T")
P = TypeVar("P")


def copy_function(f: Callable[T, P]) -> Callable[T, P]:
    """Based on https://stackoverflow.com/a/30714299"""
    g = types.FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    # in case f was given attrs (note this dict is a shallow copy):
    g.__dict__.update(f.__dict__)
    return g


class RetrieverMetaclass(ABCMeta):
    """This metaclass is used to copy the docstring from the
    `get_search_results` method, instantiated in all subclasses,
    to the `search` method in the base class.
    """

    def __new__(
        meta, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> type:
        if "search" in attrs:
            # search method was explicitly overridden, do nothing
            return type.__new__(meta, name, bases, attrs)
        # otherwise, we copy the signature and doc of the get_search_results
        # method to a copy of the search method
        get_search_results_method = attrs.get("get_search_results")
        search_method = None
        for b in bases:
            search_method = getattr(b, "search", None)
            if search_method is not None:
                break
        if search_method and get_search_results_method:
            new_search_method = copy_function(search_method)
            new_search_method.__doc__ = get_search_results_method.__doc__
            new_search_method.__signature__ = inspect.signature(  # type: ignore
                get_search_results_method
            )
            attrs["search"] = new_search_method
        return type.__new__(meta, name, bases, attrs)


class Retriever(ABC, metaclass=RetrieverMetaclass):
    """
    Abstract class for Neo4j retrievers
    """

    index_name: str
    VERIFY_NEO4J_VERSION = True

    def __init__(self, driver: neo4j.Driver, neo4j_database: Optional[str] = None):
        self.driver = driver
        self.neo4j_database = neo4j_database
        if self.VERIFY_NEO4J_VERSION:
            self._verify_version()

    def _get_version(self) -> tuple[tuple[int, ...], bool]:
        records, _, _ = self.driver.execute_query(
            "CALL dbms.components()", database_=self.neo4j_database
        )
        version = records[0]["versions"][0]
        # drop everything after the '-' first
        version_main, *_ = version.split("-")
        # convert each number between '.' into int
        version_tuple = tuple(map(int, version_main.split(".")))
        # if no patch version, consider it's 0
        if len(version_tuple) < 3:
            version_tuple = (*version_tuple, 0)
        return version_tuple, "aura" in version

    def _verify_version(self) -> None:
        """
        Check if the connected Neo4j database version supports vector indexing.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.18.1) that is known to support vector
        indexing. Raises a Neo4jMinVersionError if the connected Neo4j version is
        not supported.
        """
        version_tuple, is_aura = self._get_version()

        if is_aura:
            target_version = (5, 18, 0)
        else:
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
        query_result = self.driver.execute_query(
            query, {"index_name": self.index_name}, database_=self.neo4j_database
        )
        try:
            result = query_result.records[0]
            self._node_label = result["labels"][0]
            self._embedding_node_property = result["properties"][0]
            self._embedding_dimension = result["dimensions"]
        except IndexError as e:
            raise Exception(f"No index with name {self.index_name} found") from e

    def search(self, *args: Any, **kwargs: Any) -> RetrieverResult:
        """Search method. Call the `get_search_results` method that returns
        a list of `neo4j.Record`, and format them using the function returned by
        `get_result_formatter` to return `RetrieverResult`.
        """
        raw_result = self.get_search_results(*args, **kwargs)
        formatter = self.get_result_formatter()
        search_items = [formatter(record) for record in raw_result.records]
        metadata = raw_result.metadata or {}
        metadata["__retriever"] = self.__class__.__name__
        return RetrieverResult(
            items=search_items,
            metadata=metadata,
        )

    @abstractmethod
    def get_search_results(self, *args: Any, **kwargs: Any) -> RawSearchResult:
        """This method must be implemented in each child class. It will
        receive the same parameters provided to the public interface via
        the `search` method, after validation. It returns a `RawSearchResult`
        object which comprises a list of `neo4j.Record` objects and an optional
        `metadata` dictionary that can contain retriever-level information.

        Note that, even though this method is not intended to be called from
        outside the class, we make it public to make it clearer for the developers
        that it should be implemented in child classes.

        Returns:
            RawSearchResult: List of Neo4j Records and optional metadata dict
        """
        pass

    def get_result_formatter(self) -> Callable[[neo4j.Record], RetrieverResultItem]:
        """
        Returns the function to use to transform a neo4j.Record to a RetrieverResultItem.
        """
        if hasattr(self, "result_formatter"):
            return self.result_formatter or self.default_record_formatter
        return self.default_record_formatter

    def default_record_formatter(self, record: neo4j.Record) -> RetrieverResultItem:
        """
        Best effort to guess the node-to-text method. Inherited classes
        can override this method to implement custom text formatting.
        """
        return RetrieverResultItem(content=str(record), metadata=record.get("metadata"))


class ExternalRetriever(Retriever, ABC):
    """
    Abstract class for External Vector Stores
    """

    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        driver: neo4j.Driver,
        id_property_external: str,
        id_property_neo4j: str,
        neo4j_database: Optional[str] = None,
    ):
        super().__init__(driver)
        self.id_property_external = id_property_external
        self.id_property_neo4j = id_property_neo4j
        self.neo4j_database = neo4j_database

    @abstractmethod
    def get_search_results(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RawSearchResult:
        """

        Returns:
                RawSearchResult: List of Neo4j Records and optional metadata dict

        """
        pass
