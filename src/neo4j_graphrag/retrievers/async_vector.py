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

import logging
from typing import Any, Callable, Optional

import neo4j
from pydantic import ValidationError

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import (
    EmbeddingRequiredError,
    RetrieverInitializationError,
    SearchValidationError,
)
from neo4j_graphrag.filters import (
    FilterClassification,
    classify_filter_for_search,
    extract_filter_field_names,
)
from neo4j_graphrag.neo4j_queries import (
    _build_search_clause_vector_query,
    get_query_tail,
    get_search_query,
)
from neo4j_graphrag.retrievers.base import AsyncRetriever
from neo4j_graphrag.utils.version_utils import supports_search_clause
from neo4j_graphrag.types import (
    AsyncNeo4jDriverModel,
    AsyncVectorRetrieverModel,
    AsyncVectorCypherRetrieverModel,
    EmbedderModel,
    RawSearchResult,
    RetrieverResultItem,
    SearchType,
    VectorCypherSearchModel,
    VectorSearchModel,
)
from neo4j_graphrag.utils.logging import prettify

logger = logging.getLogger(__name__)


class AsyncVectorRetriever(AsyncRetriever):
    """
    Provides async retrieval method using vector search over embeddings.

    Example:

    .. code-block:: python

        import neo4j
        from neo4j_graphrag.retrievers import AsyncVectorRetriever

        driver = neo4j.AsyncGraphDatabase.driver(URI, auth=AUTH)
        retriever = AsyncVectorRetriever(driver, "vector-index-name", custom_embedder)
        await retriever.search(query_text="Find me a book about Fremen", top_k=5)

    Args:
        driver (neo4j.AsyncDriver): The Neo4j async Python driver.
        index_name (str): Vector index name.
        embedder (Optional[Embedder]): Embedder object to embed query text.
        return_properties (Optional[list[str]]): List of node properties to return.
        result_formatter (Optional[Callable[[neo4j.Record], RetrieverResultItem]]): Custom formatter.
        neo4j_database (Optional[str]): The name of the Neo4j database.

    Raises:
        RetrieverInitializationError: If validation of the input arguments fail.
    """

    def __init__(
        self,
        driver: neo4j.AsyncDriver,
        index_name: str,
        embedder: Optional[Embedder] = None,
        return_properties: Optional[list[str]] = None,
        result_formatter: Optional[
            Callable[[neo4j.Record], RetrieverResultItem]
        ] = None,
        neo4j_database: Optional[str] = None,
    ) -> None:
        try:
            driver_model = AsyncNeo4jDriverModel(driver=driver)
            embedder_model = EmbedderModel(embedder=embedder) if embedder else None
            validated_data = AsyncVectorRetrieverModel(
                driver_model=driver_model,
                index_name=index_name,
                embedder_model=embedder_model,
                return_properties=return_properties,
                result_formatter=result_formatter,
                neo4j_database=neo4j_database,
            )
        except ValidationError as e:
            raise RetrieverInitializationError(e.errors()) from e

        super().__init__(
            validated_data.driver_model.driver, validated_data.neo4j_database
        )
        self.index_name = validated_data.index_name
        self.return_properties = validated_data.return_properties
        self.embedder = (
            validated_data.embedder_model.embedder
            if validated_data.embedder_model
            else None
        )
        self.result_formatter = validated_data.result_formatter
        self._node_label = None
        self._embedding_node_property = None
        self._embedding_dimension = None
        self._filterable_properties: list[str] = []

    async def async_init(self) -> "AsyncVectorRetriever":
        """Fetch index metadata. Must be awaited after construction."""
        await self._fetch_index_infos(self.index_name)
        return self

    def default_record_formatter(self, record: neo4j.Record) -> RetrieverResultItem:
        metadata = {
            "score": record.get("score"),
            "nodeLabels": record.get("nodeLabels"),
            "id": record.get("id"),
        }
        node = record.get("node")
        return RetrieverResultItem(
            content=str(node),
            metadata=metadata,
        )

    async def get_search_results(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        effective_search_ratio: int = 1,
        filters: Optional[dict[str, Any]] = None,
    ) -> RawSearchResult:
        """Async vector search. See VectorRetriever.get_search_results for full docs."""
        try:
            validated_data = VectorSearchModel(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                effective_search_ratio=effective_search_ratio,
                filters=filters,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors()) from e

        parameters = validated_data.model_dump(exclude_none=True)
        parameters["vector_index_name"] = self.index_name
        if filters:
            del parameters["filters"]

        if query_text:
            if not self.embedder:
                raise EmbeddingRequiredError(
                    "Embedding method required for text query."
                )
            query_vector = self.embedder.embed_query(query_text)
            parameters["query_vector"] = query_vector
            del parameters["query_text"]

        use_search_clause = False
        filter_cls: Optional[FilterClassification] = None
        if supports_search_clause(self.driver, self.neo4j_database):
            if filters:
                filter_cls = classify_filter_for_search(filters, node_alias="node")
                missing = extract_filter_field_names(filters) - set(
                    self._filterable_properties
                )
                if not filter_cls.is_compatible:
                    logger.warning(
                        "Filters are not compatible with SEARCH clause "
                        "in-index filtering; falling back to procedure-based "
                        "vector search with brute-force filtering."
                    )
                elif missing:
                    logger.warning(
                        "Filter properties %s not declared as filterable on "
                        "index '%s'; falling back to procedure-based vector "
                        "search. Recreate the index with filterable_properties "
                        "to use in-index filtering.",
                        sorted(missing),
                        self.index_name,
                    )
                elif self._node_label:
                    use_search_clause = True
            else:
                if self._node_label:
                    use_search_clause = True

        if use_search_clause:
            search_query_base, search_params = _build_search_clause_vector_query(
                index_name=self.index_name,
                node_label=self._node_label or "",
                filter_classification=filter_cls,
            )
            query_tail = get_query_tail(
                return_properties=self.return_properties,
                fallback_return=(
                    f"RETURN node {{ .*, `{self._embedding_node_property}`: null }} AS node, "
                    "labels(node) AS nodeLabels, "
                    "elementId(node) AS elementId, "
                    "elementId(node) AS id, "
                    "score"
                )
                if self._embedding_node_property
                else (
                    "RETURN node, labels(node) AS nodeLabels, "
                    "elementId(node) AS elementId, "
                    "elementId(node) AS id, "
                    "score"
                ),
            )
            search_query = f"{search_query_base} {query_tail}"
        else:
            search_query, search_params = get_search_query(
                search_type=SearchType.VECTOR,
                return_properties=self.return_properties,
                node_label=self._node_label,
                embedding_node_property=self._embedding_node_property,
                embedding_dimension=self._embedding_dimension,
                filters=filters,
            )
        parameters.update(search_params)

        logger.debug("AsyncVectorRetriever Cypher parameters: %s", prettify(parameters))
        logger.debug("AsyncVectorRetriever Cypher query: %s", search_query)

        try:
            result = await self.driver.execute_query(
                search_query,
                parameters,
                database_=self.neo4j_database,
                routing_=neo4j.RoutingControl.READ,
            )
            records = result.records
        except neo4j.exceptions.ClientError as e:
            if use_search_clause and "PropertyNotFound" in str(e):
                logger.warning(
                    "SEARCH clause failed; falling back to procedure-based vector search. Error: %s",
                    e,
                )
                search_query, search_params = get_search_query(
                    search_type=SearchType.VECTOR,
                    return_properties=self.return_properties,
                    node_label=self._node_label,
                    embedding_node_property=self._embedding_node_property,
                    embedding_dimension=self._embedding_dimension,
                    filters=filters,
                )
                parameters.update(search_params)
                result = await self.driver.execute_query(
                    search_query,
                    parameters,
                    database_=self.neo4j_database,
                    routing_=neo4j.RoutingControl.READ,
                )
                records = result.records
            else:
                raise
        return RawSearchResult(
            records=records,
            metadata={"query_vector": query_vector},
        )


class AsyncVectorCypherRetriever(AsyncRetriever):
    """
    Provides async retrieval using vector similarity augmented by a Cypher query.

    Args:
        driver (neo4j.AsyncDriver): The Neo4j async Python driver.
        index_name (str): Vector index name.
        retrieval_query (str): Cypher query appended to the vector search.
        embedder (Optional[Embedder]): Embedder object to embed query text.
        result_formatter (Optional[Callable[[neo4j.Record], RetrieverResultItem]]): Custom formatter.
        neo4j_database (Optional[str]): The name of the Neo4j database.

    Raises:
        RetrieverInitializationError: If validation of the input arguments fail.
    """

    def __init__(
        self,
        driver: neo4j.AsyncDriver,
        index_name: str,
        retrieval_query: str,
        embedder: Optional[Embedder] = None,
        result_formatter: Optional[
            Callable[[neo4j.Record], RetrieverResultItem]
        ] = None,
        neo4j_database: Optional[str] = None,
    ) -> None:
        try:
            driver_model = AsyncNeo4jDriverModel(driver=driver)
            embedder_model = EmbedderModel(embedder=embedder) if embedder else None
            validated_data = AsyncVectorCypherRetrieverModel(
                driver_model=driver_model,
                index_name=index_name,
                retrieval_query=retrieval_query,
                embedder_model=embedder_model,
                result_formatter=result_formatter,
                neo4j_database=neo4j_database,
            )
        except ValidationError as e:
            raise RetrieverInitializationError(e.errors()) from e

        super().__init__(
            validated_data.driver_model.driver, validated_data.neo4j_database
        )
        self.index_name = validated_data.index_name
        self.retrieval_query = validated_data.retrieval_query
        self.embedder = (
            validated_data.embedder_model.embedder
            if validated_data.embedder_model
            else None
        )
        self.result_formatter = validated_data.result_formatter
        self._node_label = None
        self._node_embedding_property = None
        self._embedding_dimension = None
        self._filterable_properties: list[str] = []

    async def async_init(self) -> "AsyncVectorCypherRetriever":
        """Fetch index metadata. Must be awaited after construction."""
        await self._fetch_index_infos(self.index_name)
        return self

    async def get_search_results(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        effective_search_ratio: int = 1,
        query_params: Optional[dict[str, Any]] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> RawSearchResult:
        """Async vector+cypher search. See VectorCypherRetriever.get_search_results for full docs."""
        try:
            validated_data = VectorCypherSearchModel(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                effective_search_ratio=effective_search_ratio,
                query_params=query_params,
                filters=filters,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors()) from e

        parameters = validated_data.model_dump(exclude_none=True)
        parameters["vector_index_name"] = self.index_name
        if filters:
            del parameters["filters"]

        if query_text:
            if not self.embedder:
                raise EmbeddingRequiredError(
                    "Embedding method required for text query."
                )
            query_vector = self.embedder.embed_query(query_text)
            parameters["query_vector"] = query_vector
            del parameters["query_text"]

        if query_params:
            for key, value in query_params.items():
                if key not in parameters:
                    parameters[key] = value
            del parameters["query_params"]

        use_search_clause = False
        filter_cls: Optional[FilterClassification] = None
        if supports_search_clause(self.driver, self.neo4j_database):
            if filters:
                filter_cls = classify_filter_for_search(filters, node_alias="node")
                missing = extract_filter_field_names(filters) - set(
                    self._filterable_properties
                )
                if not filter_cls.is_compatible:
                    logger.warning(
                        "Filters are not compatible with SEARCH clause "
                        "in-index filtering; falling back to procedure-based "
                        "vector search with brute-force filtering."
                    )
                elif missing:
                    logger.warning(
                        "Filter properties %s not declared as filterable on "
                        "index '%s'; falling back to procedure-based vector "
                        "search.",
                        sorted(missing),
                        self.index_name,
                    )
                elif self._node_label:
                    use_search_clause = True
            else:
                if self._node_label:
                    use_search_clause = True

        if use_search_clause:
            search_query_base, search_params = _build_search_clause_vector_query(
                index_name=self.index_name,
                node_label=self._node_label or "",
                filter_classification=filter_cls,
            )
            query_tail = get_query_tail(retrieval_query=self.retrieval_query)
            search_query = f"{search_query_base} {query_tail}"
        else:
            search_query, search_params = get_search_query(
                search_type=SearchType.VECTOR,
                retrieval_query=self.retrieval_query,
                node_label=self._node_label,
                embedding_node_property=self._node_embedding_property,
                embedding_dimension=self._embedding_dimension,
                filters=filters,
            )
        parameters.update(search_params)

        logger.debug("AsyncVectorCypherRetriever Cypher parameters: %s", prettify(parameters))
        logger.debug("AsyncVectorCypherRetriever Cypher query: %s", search_query)

        try:
            result = await self.driver.execute_query(
                search_query,
                parameters,
                database_=self.neo4j_database,
                routing_=neo4j.RoutingControl.READ,
            )
            records = result.records
        except neo4j.exceptions.ClientError as e:
            if use_search_clause and "PropertyNotFound" in str(e):
                logger.warning(
                    "SEARCH clause failed; falling back to procedure-based vector search. Error: %s",
                    e,
                )
                search_query, search_params = get_search_query(
                    search_type=SearchType.VECTOR,
                    retrieval_query=self.retrieval_query,
                    node_label=self._node_label,
                    embedding_node_property=self._node_embedding_property,
                    embedding_dimension=self._embedding_dimension,
                    filters=filters,
                )
                parameters.update(search_params)
                result = await self.driver.execute_query(
                    search_query,
                    parameters,
                    database_=self.neo4j_database,
                    routing_=neo4j.RoutingControl.READ,
                )
                records = result.records
            else:
                raise
        return RawSearchResult(
            records=records,
            metadata={"query_vector": query_vector},
        )
