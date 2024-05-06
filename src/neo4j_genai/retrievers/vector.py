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
from typing import Optional, Any

from neo4j import Driver, Record
from neo4j_genai.retrievers.base import Retriever
from pydantic import ValidationError

from neo4j_genai.embedder import Embedder
from neo4j_genai.types import (
    VectorSearchRecord,
    VectorSearchModel,
    VectorCypherSearchModel,
    SearchType,
)
from neo4j_genai.neo4j_queries import get_search_query
import logging
import json

logger = logging.getLogger(__name__)


class VectorRetriever(Retriever):
    """
    Provides retrieval method using vector search over embeddings.
    If an embedder is provided, it needs to have the required Embedder type.
    """

    def __init__(
        self,
        driver: Driver,
        index_name: str,
        embedder: Optional[Embedder] = None,
        return_properties: Optional[list[str]] = None,
    ) -> None:
        super().__init__(driver)
        self.index_name = index_name
        self.return_properties = return_properties
        self.embedder = embedder

    def search(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
    ) -> list[VectorSearchRecord]:
        """Get the top_k nearest neighbor embeddings for either provided query_vector or query_text.
        See the following documentation for more details:

        - [Query a vector index](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-query)
        - [db.index.vector.queryNodes()](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_queryNodes)

        Args:
            query_vector (Optional[list[float]], optional): The vector embeddings to get the closest neighbors of. Defaults to None.
            query_text (Optional[str], optional): The text to get the closest neighbors of. Defaults to None.
            top_k (int, optional): The number of neighbors to return. Defaults to 5.

        Raises:
            ValueError: If validation of the input arguments fail.
            ValueError: If no embedder is provided.

        Returns:
            list[VectorSearchRecord]: The `top_k` neighbors found in vector search with their nodes and scores.
        """
        try:
            validated_data = VectorSearchModel(
                index_name=self.index_name,
                top_k=top_k,
                query_vector=query_vector,
                query_text=query_text,
            )
        except ValidationError as e:
            error_details = e.errors()
            raise ValueError(f"Validation failed: {error_details}")

        parameters = validated_data.model_dump(exclude_none=True)

        if query_text:
            if not self.embedder:
                raise ValueError("Embedding method required for text query.")
            query_vector = self.embedder.embed_query(query_text)
            parameters["query_vector"] = query_vector
            del parameters["query_text"]

        search_query = get_search_query(SearchType.VECTOR, self.return_properties)

        logger.debug(f"VectorRetriever Cypher parameters: {json.dumps(parameters)}")
        logger.debug(f"VectorRetriever Cypher query: {search_query}")

        records, _, _ = self.driver.execute_query(search_query, parameters)

        try:
            return [
                VectorSearchRecord(node=record["node"], score=record["score"])
                for record in records
            ]
        except ValidationError as e:
            error_details = e.errors()
            raise ValueError(
                f"Validation failed while constructing output: {error_details}"
            )


class VectorCypherRetriever(Retriever):
    """
    Provides retrieval method using vector similarity and custom Cypher query.
    If an embedder is provided, it needs to have the required Embedder type.
    """

    def __init__(
        self,
        driver: Driver,
        index_name: str,
        retrieval_query: str,
        embedder: Optional[Embedder] = None,
    ) -> None:
        super().__init__(driver)
        self.index_name = index_name
        self.retrieval_query = retrieval_query
        self.embedder = embedder

    def search(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        query_params: Optional[dict[str, Any]] = None,
    ) -> list[Record]:
        """Get the top_k nearest neighbor embeddings for either provided query_vector or query_text.
        See the following documentation for more details:

        - [Query a vector index](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-query)
        - [db.index.vector.queryNodes()](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_queryNodes)

        Args:
            query_vector (Optional[list[float]], optional): The vector embeddings to get the closest neighbors of. Defaults to None.
            query_text (Optional[str], optional): The text to get the closest neighbors of. Defaults to None.
            top_k (int, optional): The number of neighbors to return. Defaults to 5.
            query_params (Optional[dict[str, Any]], optional): Parameters for the Cypher query. Defaults to None.

        Raises:
            ValueError: If validation of the input arguments fail.
            ValueError: If no embedder is provided.

        Returns:
            list[Record]: The results of the search query
        """
        try:
            validated_data = VectorCypherSearchModel(
                index_name=self.index_name,
                top_k=top_k,
                query_vector=query_vector,
                query_text=query_text,
                query_params=query_params,
            )
        except ValidationError as e:
            raise ValueError(f"Validation failed: {e.errors()}")

        parameters = validated_data.model_dump(exclude_none=True)

        if query_text:
            if not self.embedder:
                raise ValueError("Embedding method required for text query.")
            parameters["query_vector"] = self.embedder.embed_query(query_text)
            del parameters["query_text"]

        if query_params:
            for key, value in query_params.items():
                if key not in parameters:
                    parameters[key] = value
            del parameters["query_params"]

        search_query = get_search_query(
            SearchType.VECTOR, retrieval_query=self.retrieval_query
        )

        logger.debug(
            f"VectorCypherRetriever Cypher parameters: {json.dumps(parameters)}"
        )
        logger.debug(f"VectorCypherRetriever Cypher query: {search_query}")

        records, _, _ = self.driver.execute_query(search_query, parameters)
        return records
