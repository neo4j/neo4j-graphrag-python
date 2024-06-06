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
from typing import Optional, Any, Callable

import neo4j

from neo4j_genai.exceptions import (
    RetrieverInitializationError,
    SearchValidationError,
    EmbeddingRequiredError,
)
from neo4j_genai.retrievers.base import Retriever
from pydantic import ValidationError

from neo4j_genai.embedder import Embedder
from neo4j_genai.types import (
    VectorSearchModel,
    VectorCypherSearchModel,
    SearchType,
    Neo4jDriverModel,
    EmbedderModel,
    VectorRetrieverModel,
    VectorCypherRetrieverModel,
    RawSearchResult,
    RetrieverResultItem,
)
from neo4j_genai.neo4j_queries import get_search_query
import logging

logger = logging.getLogger(__name__)


class VectorRetriever(Retriever):
    """
    Provides retrieval method using vector search over embeddings.
    If an embedder is provided, it needs to have the required Embedder type.

    Example:

    .. code-block:: python

      import neo4j
      from neo4j_genai import VectorRetriever

      driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)

      retriever = VectorRetriever(driver, "vector-index-name", custom_embedder)
      retriever.search(query_text="Find me a book about Fremen", top_k=5)

    Args:
        driver (neo4j.Driver): The Neo4j Python driver.
        index_name (str): Vector index name.
        embedder (Optional[Embedder]): Embedder object to embed query text.
        return_properties (Optional[list[str]]): List of node properties to return.

    Raises:
        RetrieverInitializationError: If validation of the input arguments fail.
    """

    def __init__(
        self,
        driver: neo4j.Driver,
        index_name: str,
        embedder: Optional[Embedder] = None,
        return_properties: Optional[list[str]] = None,
    ) -> None:
        try:
            driver_model = Neo4jDriverModel(driver=driver)
            embedder_model = EmbedderModel(embedder=embedder) if embedder else None
            validated_data = VectorRetrieverModel(
                driver_model=driver_model,
                index_name=index_name,
                embedder_model=embedder_model,
                return_properties=return_properties,
            )
        except ValidationError as e:
            raise RetrieverInitializationError(e.errors())

        super().__init__(driver)
        self.index_name = validated_data.index_name
        self.return_properties = validated_data.return_properties
        self.embedder = (
            validated_data.embedder_model.embedder
            if validated_data.embedder_model
            else None
        )
        self._node_label = None
        self._embedding_node_property = None
        self._embedding_dimension = None
        self._fetch_index_infos()

    def default_format_record(self, record: neo4j.Record) -> RetrieverResultItem:
        """
        Best effort to guess the node to text method. Inherited classes
        can override this method to implement custom text formatting.
        """
        metadata = {
            "score": record.get("score"),
        }
        node = record.get("node")
        return RetrieverResultItem(
            content=str(node),
            metadata=metadata,
        )

    def _get_search_results(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> RawSearchResult:
        """Get the top_k nearest neighbor embeddings for either provided query_vector or query_text.
        See the following documentation for more details:

        - `Query a vector index <https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-query>`_
        - `db.index.vector.queryNodes() <https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_queryNodes>`_

        Args:
            query_vector (Optional[list[float]]): The vector embeddings to get the closest neighbors of. Defaults to None.
            query_text (Optional[str]): The text to get the closest neighbors of. Defaults to None.
            top_k (int): The number of neighbors to return. Defaults to 5.
            filters (Optional[dict[str, Any]]): Filters for metadata pre-filtering. Defaults to None.

        Raises:
            SearchValidationError: If validation of the input arguments fail.
            EmbeddingRequiredError: If no embedder is provided.

        Returns:
            RawSearchResult: The results of the search query as a list of neo4j.Record and an optional metadata dict
        """
        try:
            validated_data = VectorSearchModel(
                vector_index_name=self.index_name,
                top_k=top_k,
                query_vector=query_vector,
                query_text=query_text,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors())

        parameters = validated_data.model_dump(exclude_none=True)

        if query_text:
            if not self.embedder:
                raise EmbeddingRequiredError(
                    "Embedding method required for text query."
                )
            query_vector = self.embedder.embed_query(query_text)
            parameters["query_vector"] = query_vector
            del parameters["query_text"]

        search_query, search_params = get_search_query(
            SearchType.VECTOR,
            self.return_properties,
            node_label=self._node_label,
            embedding_node_property=self._embedding_node_property,
            embedding_dimension=self._embedding_dimension,
            filters=filters,
        )
        parameters.update(search_params)

        logger.debug("VectorRetriever Cypher parameters: %s", parameters)
        logger.debug("VectorRetriever Cypher query: %s", search_query)

        records, _, _ = self.driver.execute_query(search_query, parameters)
        return RawSearchResult(records=records)


class VectorCypherRetriever(Retriever):
    """
    Provides retrieval method using vector similarity augmented by a Cypher query.
    This retriever builds on VectorRetriever.
    If an embedder is provided, it needs to have the required Embedder type.

    Example:

    .. code-block:: python

      import neo4j
      from neo4j_genai import VectorCypherRetriever

      driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)

      retrieval_query = "MATCH (node)-[:AUTHORED_BY]->(author:Author)" "RETURN author.name"
      retriever = VectorCypherRetriever(
        driver, "vector-index-name", retrieval_query, custom_embedder
      )
      retriever.search(query_text="Find me a book about Fremen", top_k=5)

    Args:
        driver (neo4j.Driver): The Neo4j Python driver.
        index_name (str): Vector index name.
        retrieval_query (str): Cypher query that gets appended.
        embedder (Optional[Embedder]): Embedder object to embed query text.
        format_record_function (Optional[Callable[[Any], Any]]): Function to transform a neo4j.Record to a RetrieverResultItem.
    """

    def __init__(
        self,
        driver: neo4j.Driver,
        index_name: str,
        retrieval_query: str,
        embedder: Optional[Embedder] = None,
        format_record_function: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        try:
            driver_model = Neo4jDriverModel(driver=driver)
            embedder_model = EmbedderModel(embedder=embedder) if embedder else None
            validated_data = VectorCypherRetrieverModel(
                driver_model=driver_model,
                index_name=index_name,
                retrieval_query=retrieval_query,
                embedder_model=embedder_model,
            )
        except ValidationError as e:
            raise RetrieverInitializationError(e.errors())

        super().__init__(driver)
        self.index_name = validated_data.index_name
        self.retrieval_query = validated_data.retrieval_query
        self.embedder = (
            validated_data.embedder_model.embedder
            if validated_data.embedder_model
            else None
        )
        self.format_record_function = format_record_function
        self._node_label = None
        self._node_embedding_property = None
        self._embedding_dimension = None
        self._fetch_index_infos()

    def _get_search_results(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        query_params: Optional[dict[str, Any]] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> RawSearchResult:
        """Get the top_k nearest neighbor embeddings for either provided query_vector or query_text.
        See the following documentation for more details:

        - `Query a vector index <https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-query>`_
        - `db.index.vector.queryNodes() <https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_queryNodes>`_

        Args:
            query_vector (Optional[list[float]]): The vector embeddings to get the closest neighbors of. Defaults to None.
            query_text (Optional[str]): The text to get the closest neighbors of. Defaults to None.
            top_k (int): The number of neighbors to return. Defaults to 5.
            query_params (Optional[dict[str, Any]]): Parameters for the Cypher query. Defaults to None.
            filters (Optional[dict[str, Any]]): Filters for metadata pre-filtering. Defaults to None.

        Raises:
            SearchValidationError: If validation of the input arguments fail.
            EmbeddingRequiredError: If no embedder is provided.

        Returns:
            RawSearchResult: The results of the search query as a list of neo4j.Record and an optional metadata dict
        """
        try:
            validated_data = VectorCypherSearchModel(
                vector_index_name=self.index_name,
                top_k=top_k,
                query_vector=query_vector,
                query_text=query_text,
                query_params=query_params,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors())

        parameters = validated_data.model_dump(exclude_none=True)

        if query_text:
            if not self.embedder:
                raise EmbeddingRequiredError(
                    "Embedding method required for text query."
                )
            parameters["query_vector"] = self.embedder.embed_query(query_text)
            del parameters["query_text"]

        if query_params:
            for key, value in query_params.items():
                if key not in parameters:
                    parameters[key] = value
            del parameters["query_params"]

        search_query, search_params = get_search_query(
            SearchType.VECTOR,
            retrieval_query=self.retrieval_query,
            node_label=self._node_label,
            embedding_node_property=self._node_embedding_property,
            embedding_dimension=self._embedding_dimension,
            filters=filters,
        )
        parameters.update(search_params)

        logger.debug("VectorCypherRetriever Cypher parameters: %s", parameters)
        logger.debug("VectorCypherRetriever Cypher query: %s", search_query)

        records, _, _ = self.driver.execute_query(search_query, parameters)
        return RawSearchResult(
            records=records,
        )
