import neo4j

from typing import List, Dict, Any, Optional
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import CypherSyntaxError

from abc import ABC, abstractmethod


class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""


class GenAIClient:
    def __init__(self, driver: Driver, embeddings: Optional[Embeddings]) -> None:
        # Verify if the version supports vector index
        self._verify_version(driver)
        self.embeddings = embeddings if embeddings else None

    def _verify_version(self, driver: Driver) -> None:
        """
        Check if the connected Neo4j database version supports vector indexing.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.11.0) that is known to support vector
        indexing. Raises a ValueError if the connected Neo4j version is
        not supported.
        """
        version = self.database_query(driver, "CALL dbms.components()")[0]["versions"][
            0
        ]
        if "aura" in version:
            version_tuple = (*tuple(map(int, version.split("-")[0].split("."))), 0)
        else:
            version_tuple = tuple(map(int, version.split(".")))

        target_version = (5, 11, 0)

        if version_tuple < target_version:
            raise ValueError(
                "Version index is only supported in Neo4j version 5.11 or greater"
            )

    def database_query(
        self, driver: Driver, query: str, params: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        This method sends a Cypher query to the connected Neo4j database
        and returns the results as a list of dictionaries.

        Args:
            query (str): The Cypher query to execute.
            params (Dict, optional): Dictionary of query parameters. Defaults to {}.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing the query results.
        """
        params = params or {}
        # TODO: how do we pass this database variable
        with driver.session(database="neo4j") as session:
            try:
                data = session.run(query, params)
                return [r.data() for r in data]
            except CypherSyntaxError as e:
                raise ValueError(f"Cypher Statement is not valid\n{e}")

    def create_index(
        self,
        driver: Driver,
        name: str,
        label: str,
        property: str,
        dimensions: int,
        similarity_fn: str,
    ) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new vector index in Neo4j.
        """
        index_query = (
            "CALL db.index.vector.createNodeIndex("
            "$name,"
            "$label,"
            "$property,"
            "toInteger($dimensions),"
            "$similarity_fn )"
        )

        parameters = {
            "name": name,
            "node_label": label,
            "property": property,
            "dimensions": dimensions,
            "similarity_fn": similarity_fn,
        }
        self.database_query(driver, index_query, params=parameters)

    def similarity_search(
        self,
        driver: Driver,
        name: str,
        query_vector: Optional[List[float]],
        query_text: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Performs the similarity search
        """
        if not ((query_vector is not None) ^ (query_text is not None)):
            raise ValueError("You must provide one of query_vector or query_text.")

        if query_vector:
            parameters = {
                "index_name": name,
                "top_k": top_k,
                "vector": query_vector,
            }

        if query_text:
            # TODO: do we need to validate embeddings? Normalizing etc.
            if self.embeddings:
                vector_embedding = self.embeddings.embed_query(query_text)
                parameters = {
                    "index_name": name,
                    "top_k": top_k,
                    "vector": vector_embedding,
                }
            else:
                raise ValueError(
                    "Embeddings required in definition to perform search for query_text"
                )

        db_query_string = "CALL db.index.vector.queryNodes($index_name, $top_k, $vector) YIELD node, score"
        return self.database_query(driver, db_query_string, params=parameters)
