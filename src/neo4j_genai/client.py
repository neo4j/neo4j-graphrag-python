from typing import List, Dict, Any, Optional
from neo4j import Driver
from neo4j.exceptions import CypherSyntaxError
from neo4j_genai.embeddings import Embeddings
from neo4j_genai.types import CreateIndexModel, SimilaritySearchModel
from pydantic import ValidationError


class GenAIClient:
    def __init__(self, driver: Driver, embeddings: Optional[Embeddings] = None) -> None:
        # Verify if the version supports vector index
        self.driver = driver
        self._verify_version()
        self.embeddings = embeddings

    def _verify_version(self) -> None:
        """
        Check if the connected Neo4j database version supports vector indexing.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.11.0) that is known to support vector
        indexing. Raises a ValueError if the connected Neo4j version is
        not supported.
        """
        version = self.database_query("CALL dbms.components()")[0]["versions"][0]
        if "aura" in version:
            version_tuple = (*tuple(map(int, version.split("-")[0].split("."))), 0)
        else:
            version_tuple = tuple(map(int, version.split(".")))

        target_version = (5, 11, 0)

        if version_tuple < target_version:
            raise ValueError(
                "Version index is only supported in Neo4j version 5.11 or greater"
            )

    def database_query(self, query: str, params: Dict = {}) -> List[Dict[str, Any]]:
        """
        This method sends a Cypher query to the connected Neo4j database
        and returns the results as a list of dictionaries.

        Args:
            query (str): The Cypher query to execute.
            params (Dict, optional): Dictionary of query parameters. Defaults to {}.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing the query results.
        """
        with self.driver.session() as session:
            try:
                data = session.run(query, params)
                return [r.data() for r in data]
            except CypherSyntaxError as e:
                raise ValueError(f"Cypher Statement is not valid\n{e}")

    def create_index(
        self,
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
        index_data = {
            "name": name,
            "label": label,
            "property": property,
            "dimensions": dimensions,
            "similarity_fn": similarity_fn,
        }
        try:
            index_data = CreateIndexModel(**index_data)
        except ValidationError as e:
            raise ValueError(f"Error for inputs to create_index {str(e)}")

        query = (
            "CALL db.index.vector.createNodeIndex("
            "$name,"
            "$label,"
            "$property,"
            "toInteger($dimensions),"
            "$similarity_fn )"
        )
        self.database_query(query, params=index_data.dict())

    def drop_index(self, name: str) -> None:
        """
        This method constructs a Cypher query and executes it
        to drop a vector index in Neo4j.
        """
        query = "DROP INDEX $name"
        parameters = {
            "name": name,
        }
        self.database_query(query, params=parameters)

    def similarity_search(
        self,
        name: str,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Performs the similarity search
        """
        try:
            validated_data = SimilaritySearchModel(
                index_name=name,
                top_k=top_k,
                query_vector=query_vector,
                query_text=query_text,
            )
        except ValidationError as e:
            error_details = e.errors()
            raise ValueError(f"Validation failed: {error_details}")

        if query_text:
            if not self.embeddings:
                raise ValueError("Embedding method required for text query.")
            query_vector = self.embeddings.embed_query(query_text)

        parameters = validated_data.dict(exclude_none=True)
        db_query_string = "CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector) YIELD node, score"
        return self.database_query(db_query_string, params=parameters)
