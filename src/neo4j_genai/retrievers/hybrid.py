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

import neo4j
from pydantic import ValidationError

from neo4j_genai.embedder import Embedder
from neo4j_genai.retrievers.base import Retriever
from neo4j_genai.types import (
    HybridSearchModel,
    SearchType,
    HybridCypherSearchModel,
    Neo4jDriverModel,
    EmbedderModel,
    HybridRetrieverModel,
    HybridCypherRetrieverModel,
)
from neo4j_genai.neo4j_queries import get_search_query
import logging

logger = logging.getLogger(__name__)


class HybridRetriever(Retriever):
    def __init__(
        self,
        driver: neo4j.Driver,
        vector_index_name: str,
        fulltext_index_name: str,
        embedder: Optional[Embedder] = None,
        return_properties: Optional[list[str]] = None,
    ) -> None:
        try:
            driver_model = Neo4jDriverModel(driver=driver)
            embedder_model = EmbedderModel(embedder=embedder) if embedder else None
            validated_data = HybridRetrieverModel(
                driver_model=driver_model,
                vector_index_name=vector_index_name,
                fulltext_index_name=fulltext_index_name,
                embedder_model=embedder_model,
                return_properties=return_properties,
            )
        except ValidationError as e:
            raise ValueError(f"Validation failed: {e.errors()}")

        super().__init__(validated_data.driver_model.driver)
        self.vector_index_name = validated_data.vector_index_name
        self.fulltext_index_name = validated_data.fulltext_index_name
        self.return_properties = validated_data.return_properties
        self.embedder = (
            validated_data.embedder_model.embedder
            if validated_data.embedder_model
            else None
        )

    def search(
        self,
        query_text: str,
        query_vector: Optional[list[float]] = None,
        top_k: int = 5,
    ) -> list[neo4j.Record]:
        """Get the top_k nearest neighbor embeddings for either provided query_vector or query_text.
        Both query_vector and query_text can be provided.
        If query_vector is provided, then it will be preferred over the embedded query_text
        for the vector search.
        See the following documentation for more details:
        - [Query a vector index](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-query)
        - [db.index.vector.queryNodes()](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_queryNodes)
        - [db.index.fulltext.queryNodes()](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_fulltext_querynodes)
        Args:
            query_text (str): The text to get the closest neighbors of.
            query_vector (Optional[list[float]], optional): The vector embeddings to get the closest neighbors of. Defaults to None.
            top_k (int, optional): The number of neighbors to return. Defaults to 5.
        Raises:
            ValueError: If validation of the input arguments fail.
            ValueError: If no embedder is provided.
        Returns:
            list[neo4j.Record]: The results of the search query
        """
        try:
            validated_data = HybridSearchModel(
                vector_index_name=self.vector_index_name,
                fulltext_index_name=self.fulltext_index_name,
                top_k=top_k,
                query_vector=query_vector,
                query_text=query_text,
            )
        except ValidationError as e:
            raise ValueError(f"Validation failed: {e.errors()}")

        parameters = validated_data.model_dump(exclude_none=True)

        if query_text and not query_vector:
            if not self.embedder:
                raise ValueError("Embedding method required for text query.")
            query_vector = self.embedder.embed_query(query_text)
            parameters["query_vector"] = query_vector

        search_query, _ = get_search_query(SearchType.HYBRID, self.return_properties)

        logger.debug("HybridRetriever Cypher parameters: %s", parameters)
        logger.debug("HybridRetriever Cypher query: %s", search_query)

        records, _, _ = self.driver.execute_query(search_query, parameters)
        return records


class HybridCypherRetriever(Retriever):
    def __init__(
        self,
        driver: neo4j.Driver,
        vector_index_name: str,
        fulltext_index_name: str,
        retrieval_query: str,
        embedder: Optional[Embedder] = None,
    ) -> None:
        try:
            driver_model = Neo4jDriverModel(driver=driver)
            embedder_model = EmbedderModel(embedder=embedder) if embedder else None
            validated_data = HybridCypherRetrieverModel(
                driver_model=driver_model,
                vector_index_name=vector_index_name,
                fulltext_index_name=fulltext_index_name,
                retrieval_query=retrieval_query,
                embedder_model=embedder_model,
            )
        except ValidationError as e:
            raise ValueError(f"Validation failed: {e.errors()}")

        super().__init__(validated_data.driver_model.driver)
        self.vector_index_name = validated_data.vector_index_name
        self.fulltext_index_name = validated_data.fulltext_index_name
        self.retrieval_query = validated_data.retrieval_query
        self.embedder = (
            validated_data.embedder_model.embedder
            if validated_data.embedder_model
            else None
        )

    def search(
        self,
        query_text: str,
        query_vector: Optional[list[float]] = None,
        top_k: int = 5,
        query_params: Optional[dict[str, Any]] = None,
    ) -> list[neo4j.Record]:
        """Get the top_k nearest neighbor embeddings for either provided query_vector or query_text.
        Both query_vector and query_text can be provided.
        If query_vector is provided, then it will be preferred over the embedded query_text
        for the vector search.
        See the following documentation for more details:
        - [Query a vector index](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-query)
        - [db.index.vector.queryNodes()](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_queryNodes)
        - [db.index.fulltext.queryNodes()](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_fulltext_querynodes)
        Args:
            query_text (str): The text to get the closest neighbors of.
            query_vector (Optional[list[float]], optional): The vector embeddings to get the closest neighbors of. Defaults to None.
            top_k (int, optional): The number of neighbors to return. Defaults to 5.
            query_params (Optional[dict[str, Any]], optional): Parameters for the Cypher query. Defaults to None.
        Raises:
            ValueError: If validation of the input arguments fail.
            ValueError: If no embedder is provided.
        Returns:
            list[neo4j.Record]: The results of the search query
        """
        try:
            validated_data = HybridCypherSearchModel(
                vector_index_name=self.vector_index_name,
                fulltext_index_name=self.fulltext_index_name,
                top_k=top_k,
                query_vector=query_vector,
                query_text=query_text,
                query_params=query_params,
            )
        except ValidationError as e:
            raise ValueError(f"Validation failed: {e.errors()}")

        parameters = validated_data.model_dump(exclude_none=True)

        if query_text and not query_vector:
            if not self.embedder:
                raise ValueError("Embedding method required for text query.")
            query_vector = self.embedder.embed_query(query_text)
            parameters["query_vector"] = query_vector

        if query_params:
            for key, value in query_params.items():
                if key not in parameters:
                    parameters[key] = value
            del parameters["query_params"]

        search_query, _ = get_search_query(
            SearchType.HYBRID, retrieval_query=self.retrieval_query
        )

        logger.debug("HybridCypherRetriever Cypher parameters: %s", parameters)
        logger.debug("HybridCypherRetriever Cypher query: %s", search_query)

        records, _, _ = self.driver.execute_query(search_query, parameters)
        return records
