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
from typing import Any, Callable, Optional, Union

import neo4j
from pydantic import ValidationError

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import (
    EmbeddingRequiredError,
    RetrieverInitializationError,
    SearchValidationError,
    SearchQueryParseError,
)
from neo4j_graphrag.neo4j_queries import (
    _build_hybrid_search_clause_query,
    _build_hybrid_search_clause_query_linear,
    get_query_tail,
    get_search_query,
)
from neo4j_graphrag.retrievers.base import AsyncRetriever
from neo4j_graphrag.utils.version_utils import supports_search_clause
from neo4j_graphrag.types import (
    AsyncNeo4jDriverModel,
    AsyncHybridRetrieverModel,
    AsyncHybridCypherRetrieverModel,
    EmbedderModel,
    HybridCypherSearchModel,
    HybridSearchModel,
    HybridSearchRanker,
    RawSearchResult,
    RetrieverResultItem,
    SearchType,
)
from neo4j_graphrag.utils.logging import prettify

logger = logging.getLogger(__name__)


class AsyncHybridRetriever(AsyncRetriever):
    """
    Provides async retrieval using combination of vector search and fulltext search.

    Args:
        driver (neo4j.AsyncDriver): The Neo4j async Python driver.
        vector_index_name (str): Vector index name.
        fulltext_index_name (str): Fulltext index name.
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
        vector_index_name: str,
        fulltext_index_name: str,
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
            validated_data = AsyncHybridRetrieverModel(
                driver_model=driver_model,
                vector_index_name=vector_index_name,
                fulltext_index_name=fulltext_index_name,
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
        self.vector_index_name = validated_data.vector_index_name
        self.fulltext_index_name = validated_data.fulltext_index_name
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

    async def async_init(self) -> "AsyncHybridRetriever":
        """Fetch index metadata. Must be awaited after construction."""
        await self._fetch_index_infos(self.vector_index_name)
        return self

    def default_record_formatter(self, record: neo4j.Record) -> RetrieverResultItem:
        metadata = {"score": record.get("score")}
        node = record.get("node")
        return RetrieverResultItem(content=str(node), metadata=metadata)

    async def get_search_results(
        self,
        query_text: str,
        query_vector: Optional[list[float]] = None,
        top_k: int = 5,
        effective_search_ratio: int = 1,
        ranker: Union[str, HybridSearchRanker] = HybridSearchRanker.NAIVE,
        alpha: Optional[float] = None,
    ) -> RawSearchResult:
        """Async hybrid search. See HybridRetriever.get_search_results for full docs."""
        try:
            validated_data = HybridSearchModel(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                effective_search_ratio=effective_search_ratio,
                ranker=ranker,
                alpha=alpha,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors()) from e

        parameters = validated_data.model_dump(exclude_none=True)
        parameters["vector_index_name"] = self.vector_index_name
        parameters["fulltext_index_name"] = self.fulltext_index_name

        if query_text and not query_vector:
            if not self.embedder:
                raise EmbeddingRequiredError(
                    "Embedding method required for text query."
                )
            query_vector = self.embedder.embed_query(query_text)
            parameters["query_vector"] = query_vector

        use_search_clause = False
        if supports_search_clause(self.driver, self.neo4j_database):
            if self._node_label:
                use_search_clause = True

        if use_search_clause:
            if validated_data.ranker == HybridSearchRanker.LINEAR and validated_data.alpha:
                search_query_base = _build_hybrid_search_clause_query_linear(
                    vector_index_name=self.vector_index_name,
                    fulltext_index_name=self.fulltext_index_name,
                    node_label=self._node_label or "",
                )
            else:
                search_query_base = _build_hybrid_search_clause_query(
                    vector_index_name=self.vector_index_name,
                    fulltext_index_name=self.fulltext_index_name,
                    node_label=self._node_label or "",
                )
            query_tail = get_query_tail(
                return_properties=self.return_properties,
                fallback_return=(
                    f"RETURN node {{ .*, `{self._embedding_node_property}`: null }} AS node, "
                    "labels(node) AS nodeLabels, elementId(node) AS elementId, "
                    "elementId(node) AS id, score"
                )
                if self._embedding_node_property
                else (
                    "RETURN node, labels(node) AS nodeLabels, "
                    "elementId(node) AS elementId, elementId(node) AS id, score"
                ),
            )
            search_query = f"{search_query_base} {query_tail}"
        else:
            search_query, _ = get_search_query(
                search_type=SearchType.HYBRID,
                return_properties=self.return_properties,
                embedding_node_property=self._embedding_node_property,
                neo4j_version_is_5_23_or_above=getattr(self, "neo4j_version_is_5_23_or_above", True),
                ranker=validated_data.ranker,
                alpha=validated_data.alpha,
            )

        if "ranker" in parameters:
            del parameters["ranker"]

        logger.debug("AsyncHybridRetriever Cypher parameters: %s", prettify(parameters))
        logger.debug("AsyncHybridRetriever Cypher query: %s", search_query)

        try:
            result = await self.driver.execute_query(
                search_query,
                parameters,
                database_=self.neo4j_database,
                routing_=neo4j.RoutingControl.READ,
            )
            records = result.records
        except neo4j.exceptions.ClientError as e:
            if "org.apache.lucene.queryparser.classic.ParseException" in str(e):
                raise SearchQueryParseError(
                    f"Invalid Lucene query generated from query_text: {query_text}"
                ) from e
            raise
        return RawSearchResult(
            records=records,
            metadata={"query_vector": query_vector},
        )


class AsyncHybridCypherRetriever(AsyncRetriever):
    """
    Provides async retrieval using hybrid search augmented by a Cypher query.

    Args:
        driver (neo4j.AsyncDriver): The Neo4j async Python driver.
        vector_index_name (str): Vector index name.
        fulltext_index_name (str): Fulltext index name.
        retrieval_query (str): Cypher query appended to the hybrid search.
        embedder (Optional[Embedder]): Embedder object to embed query text.
        result_formatter (Optional[Callable[[neo4j.Record], RetrieverResultItem]]): Custom formatter.
        neo4j_database (Optional[str]): The name of the Neo4j database.

    Raises:
        RetrieverInitializationError: If validation of the input arguments fail.
    """

    def __init__(
        self,
        driver: neo4j.AsyncDriver,
        vector_index_name: str,
        fulltext_index_name: str,
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
            validated_data = AsyncHybridCypherRetrieverModel(
                driver_model=driver_model,
                vector_index_name=vector_index_name,
                fulltext_index_name=fulltext_index_name,
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
        self.vector_index_name = validated_data.vector_index_name
        self.fulltext_index_name = validated_data.fulltext_index_name
        self.retrieval_query = validated_data.retrieval_query
        self.embedder = (
            validated_data.embedder_model.embedder
            if validated_data.embedder_model
            else None
        )
        self.result_formatter = validated_data.result_formatter
        self._node_label = None
        self._embedding_node_property = None
        self._embedding_dimension = None

    async def async_init(self) -> "AsyncHybridCypherRetriever":
        """Fetch index metadata. Must be awaited after construction."""
        await self._fetch_index_infos(self.vector_index_name)
        return self

    async def get_search_results(
        self,
        query_text: str,
        query_vector: Optional[list[float]] = None,
        top_k: int = 5,
        effective_search_ratio: int = 1,
        query_params: Optional[dict[str, Any]] = None,
        ranker: Union[str, HybridSearchRanker] = HybridSearchRanker.NAIVE,
        alpha: Optional[float] = None,
    ) -> RawSearchResult:
        """Async hybrid+cypher search. See HybridCypherRetriever.get_search_results for full docs."""
        try:
            validated_data = HybridCypherSearchModel(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                effective_search_ratio=effective_search_ratio,
                ranker=ranker,
                alpha=alpha,
                query_params=query_params,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors()) from e

        parameters = validated_data.model_dump(exclude_none=True)
        parameters["vector_index_name"] = self.vector_index_name
        parameters["fulltext_index_name"] = self.fulltext_index_name

        if query_text and not query_vector:
            if not self.embedder:
                raise EmbeddingRequiredError(
                    "Embedding method required for text query."
                )
            query_vector = self.embedder.embed_query(query_text)
            parameters["query_vector"] = query_vector

        if query_params:
            for key, value in query_params.items():
                if key not in parameters:
                    parameters[key] = value
            del parameters["query_params"]

        use_search_clause = False
        if supports_search_clause(self.driver, self.neo4j_database):
            if self._node_label:
                use_search_clause = True

        if use_search_clause:
            if validated_data.ranker == HybridSearchRanker.LINEAR and validated_data.alpha:
                search_query_base = _build_hybrid_search_clause_query_linear(
                    vector_index_name=self.vector_index_name,
                    fulltext_index_name=self.fulltext_index_name,
                    node_label=self._node_label or "",
                )
            else:
                search_query_base = _build_hybrid_search_clause_query(
                    vector_index_name=self.vector_index_name,
                    fulltext_index_name=self.fulltext_index_name,
                    node_label=self._node_label or "",
                )
            query_tail = get_query_tail(retrieval_query=self.retrieval_query)
            search_query = f"{search_query_base} {query_tail}"
        else:
            search_query, _ = get_search_query(
                search_type=SearchType.HYBRID,
                retrieval_query=self.retrieval_query,
                neo4j_version_is_5_23_or_above=getattr(self, "neo4j_version_is_5_23_or_above", True),
                ranker=validated_data.ranker,
                alpha=validated_data.alpha,
            )

        if "ranker" in parameters:
            del parameters["ranker"]

        logger.debug("AsyncHybridCypherRetriever Cypher parameters: %s", prettify(parameters))
        logger.debug("AsyncHybridCypherRetriever Cypher query: %s", search_query)

        try:
            result = await self.driver.execute_query(
                search_query,
                parameters,
                database_=self.neo4j_database,
                routing_=neo4j.RoutingControl.READ,
            )
            records = result.records
        except neo4j.exceptions.ClientError as e:
            if "org.apache.lucene.queryparser.classic.ParseException" in str(e):
                raise SearchQueryParseError(
                    f"Invalid Lucene query generated from query_text: {query_text}"
                ) from e
            raise
        return RawSearchResult(
            records=records,
            metadata={"query_vector": query_vector},
        )
