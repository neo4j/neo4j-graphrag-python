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

from pydantic import ValidationError

from neo4j_genai.exceptions import RetrieverInitializationError, SearchValidationError
from neo4j_genai.retrievers.base import ExternalRetriever
from neo4j_genai.embedder import Embedder
from neo4j_genai.retrievers.external.weaviate.types import (
    WeaviateModel,
    WeaviateNeo4jRetrieverModel,
    WeaviateNeo4jSearchModel,
)
import weaviate.classes as wvc
from weaviate.client import WeaviateClient
import neo4j
import logging
from neo4j_genai.neo4j_queries import get_query_tail
from neo4j_genai.types import Neo4jDriverModel, EmbedderModel

logger = logging.getLogger(__name__)


class WeaviateNeo4jRetriever(ExternalRetriever):
    def __init__(
        self,
        driver: neo4j.Driver,
        client: WeaviateClient,
        collection: str,
        id_property_external: str,
        id_property_neo4j: str,
        embedder: Optional[Embedder] = None,
        return_properties: Optional[list[str]] = None,
        retrieval_query: Optional[str] = None,
    ):
        try:
            driver_model = Neo4jDriverModel(driver=driver)
            weaviate_model = WeaviateModel(client=client)
            embedder_model = EmbedderModel(embedder=embedder) if embedder else None
            validated_data = WeaviateNeo4jRetrieverModel(
                driver_model=driver_model,
                client_model=weaviate_model,
                collection=collection,
                id_property_external=id_property_external,
                id_property_neo4j=id_property_neo4j,
                embedder_model=embedder_model,
                return_properties=return_properties,
                retrieval_query=retrieval_query,
            )
        except ValidationError as e:
            raise RetrieverInitializationError(e.errors())

        super().__init__(id_property_external, id_property_neo4j)
        self.driver = validated_data.driver_model.driver
        self.client = validated_data.client_model.client
        collection = validated_data.collection
        self.search_collection = self.client.collections.get(collection)
        self.embedder = (
            validated_data.embedder_model.embedder
            if validated_data.embedder_model
            else None
        )
        self.return_properties = validated_data.return_properties
        self.retrieval_query = validated_data.retrieval_query

    def search(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[neo4j.Record]:
        """Get the top_k nearest neighbor embeddings using Weaviate for either provided query_vector or query_text.
        Both query_vector and query_text can be provided.
        If query_vector is provided, then it will be preferred over the embedded query_text
        for the vector search.
        If query_text is provided, then it will check if an embedder is provided and use it to generate the query_vector.
        If no embedder is provided, then it will assume that the vectorizer is used in Weaviate.

        See the following documentation for more details:
        - [Query a vector index](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-query)
        - [db.index.vector.queryNodes()](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_queryNodes)
        - [db.index.fulltext.queryNodes()](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_fulltext_querynodes)
        Args:
            query_text (str): The text to get the closest neighbors of.
            query_vector (Optional[list[float]], optional): The vector embeddings to get the closest neighbors of. Defaults to None.
            top_k (int, optional): The number of neighbors to return. Defaults to 5.
            weaviate_filters (Optional[_Filters], optional): The filters to apply to the search query in Weaviate. Defaults to None.
        Raises:
            SearchValidationError: If validation of the input arguments fail.
        Returns:
            list[neo4j.Record]: The results of the search query
        """

        weaviate_filters = kwargs.get("weaviate_filters")

        try:
            validated_data = WeaviateNeo4jSearchModel(
                top_k=top_k,
                query_vector=query_vector,
                query_text=query_text,
                weaviate_filters=weaviate_filters,
            )
            query_text = validated_data.query_text
            query_vector = validated_data.query_vector
            top_k = validated_data.top_k
            weaviate_filters = validated_data.weaviate_filters
        except ValidationError as e:
            raise SearchValidationError(e.errors())

        # If we want to use a local embedder, we still want to call the near_vector method
        # so we want to create the vector as early as possible here
        if query_text:
            if self.embedder:
                query_vector = self.embedder.embed_query(query_text)
                logger.debug("Locally generated query vector: %s", query_vector)
            else:
                logger.debug(
                    "No embedder provided, assuming vectorizer is used in Weaviate."
                )
        weaviate_filters = kwargs.get("weaviate_filters")

        if query_vector:
            response = self.search_collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=weaviate_filters,
                return_metadata=wvc.query.MetadataQuery(certainty=True),
            )
            logger.debug("Weaviate query vector: %s", query_vector)
            logger.debug("Response: %s", response)
        else:
            response = self.search_collection.query.near_text(
                query=query_text,
                limit=top_k,
                filters=weaviate_filters,
                return_metadata=wvc.query.MetadataQuery(certainty=True),
            )
            logger.debug("Query text: %s", query_text)
            logger.debug("Response: %s", response)

        result_tuples = [
            [f"{o.properties[self.id_property_external]}", o.metadata.certainty or 0.0]
            for o in response.objects
        ]

        search_query = get_match_query(
            return_properties=self.return_properties,
            retrieval_query=self.retrieval_query,
        )

        parameters = {
            "match_params": result_tuples,
            "id_property": self.id_property_neo4j,
        }

        logger.debug("Weaviate Store Cypher parameters: %s", parameters)
        logger.debug("Weaviate Store Cypher query: %s", search_query)

        records, _, _ = self.driver.execute_query(search_query, parameters)

        return records


def get_match_query(
    return_properties: Optional[list[str]] = None, retrieval_query: Optional[str] = None
) -> str:
    match_query = (
        "UNWIND $match_params AS match_param "
        "WITH match_param[0] AS match_id_value, match_param[1] AS score "
        "MATCH (node) "
        "WHERE node[$id_property] = match_id_value "
    )
    return match_query + get_query_tail(
        return_properties=return_properties,
        retrieval_query=retrieval_query,
        fallback_return="RETURN node, score",
    )
