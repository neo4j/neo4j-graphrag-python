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

import logging
from typing import Any, Callable, Optional

import neo4j
from pinecone import Pinecone
from pydantic import ValidationError

from neo4j_genai.embedder import Embedder
from neo4j_genai.retrievers.base import ExternalRetriever
from neo4j_genai.retrievers.external.utils import get_match_query
from neo4j_genai.types import (
    EmbedderModel,
    Neo4jDriverModel,
    PineconeClientModel,
    PineconeNeo4jRetrieverModel,
    PineconeSearchModel,
    RawSearchResult,
)

logger = logging.getLogger(__name__)


class PineconeNeo4jRetriever(ExternalRetriever):
    def __init__(
        self,
        driver: neo4j.Driver,
        client: Pinecone,
        index_name: str,
        id_property_neo4j: str,
        embedder: Optional[Embedder] = None,
        return_properties: Optional[list[str]] = None,
        retrieval_query: Optional[str] = None,
        format_record_function: Optional[Callable] = None,
    ):
        try:
            driver_model = Neo4jDriverModel(driver=driver)
            client_model = PineconeClientModel(client=client)
            embedder_model = EmbedderModel(embedder=embedder) if embedder else None
            validated_data = PineconeNeo4jRetrieverModel(
                driver_model=driver_model,
                client_model=client_model,
                index_name=index_name,
                id_property_neo4j=id_property_neo4j,
                embedder_model=embedder_model,
                return_properties=return_properties,
                retrieval_query=retrieval_query,
                format_record_function=format_record_function,
            )
        except ValidationError as e:
            raise ValueError(f"Validation failed: {e.errors()}")

        super().__init__(
            driver=driver,
            id_property_external="id",
            id_property_neo4j=validated_data.id_property_neo4j,
        )
        self.driver = validated_data.driver_model.driver
        self.client = validated_data.client_model.client
        self.index_name = validated_data.index_name
        self.index = self.client.Index(index_name)
        self.embedder = (
            validated_data.embedder_model.embedder
            if validated_data.embedder_model
            else None
        )
        self.return_properties = validated_data.return_properties
        self.retrieval_query = validated_data.retrieval_query

    def _get_search_results(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        pinecone_filter: Optional[dict[str, Any]] = None,
    ) -> RawSearchResult:
        """Get the top_k nearest neighbor embeddings using Pinecone for either provided query_vector or query_text.
        Both query_vector and query_text can be provided.
        If query_vector is provided, then it will be preferred over the embedded query_text
        for the vector search.
        If query_text is provided, then it will check if an embedder is provided and use it to generate the query_vector.

        See the following documentation for more details:
        - [Query a vector index](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-query)
        - [db.index.vector.queryNodes()](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_queryNodes)
        - [db.index.fulltext.queryNodes()](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_fulltext_querynodes)
        Args:
            query_text (str): The text to get the closest neighbors of.
            query_vector (Optional[list[float]], optional): The vector embeddings to get the closest neighbors of. Defaults to None.
            top_k (int, optional): The number of neighbors to return. Defaults to 5.
            pinecone_filter (Optional[dict[str, Any]], optional): The filters to apply to the search query in Pinecone. Defaults to None.
        Raises:
            ValueError: If validation of the input arguments fail.
        Returns:
            RawSearchResult: The results of the search query as a list of neo4j.Record and an optional metadata dict
        """

        try:
            validated_data = PineconeSearchModel(
                vector_index_name=self.index_name,
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                pinecone_filter=pinecone_filter,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors())

        if validated_data.query_text:
            if self.embedder:
                query_vector = self.embedder.embed_query(validated_data.query_text)
                logger.debug("Locally generated query vector: %s", query_vector)
            else:
                logger.error("No embedder provided for query_text.")
                raise RuntimeError("No embedder provided for query_text.")

        response = self.index.query(
            vector=query_vector,
            top_k=validated_data.top_k,
            filter=validated_data.pinecone_filter,
        )

        result_tuples = [
            [f"{o[self.id_property_external]}", o["score"] or 0.0]
            for o in response["matches"]
        ]

        search_query = get_match_query(
            return_properties=self.return_properties,
            retrieval_query=self.retrieval_query,
        )

        parameters = {
            "match_params": result_tuples,
            "id_property": self.id_property_neo4j,
        }

        logger.debug("Pinecone Store Cypher parameters: %s", parameters)
        logger.debug("Pinecone Store Cypher query: %s", search_query)

        records, _, _ = self.driver.execute_query(search_query, parameters)

        return RawSearchResult(records=records)
