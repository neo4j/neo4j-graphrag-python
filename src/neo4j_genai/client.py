from typing import List, Dict, Any, Optional
from neo4j import Driver
from neo4j.exceptions import CypherSyntaxError
from neo4j_genai.embeddings import Embeddings
from neo4j_genai.types import CreateIndexModel, SimilaritySearchModel
from pydantic import ValidationError


class GenAIClient:
    def __init__(self, driver: Driver, embeddings: Optional[Embeddings] = None) -> None:
        # Verify if the version supports vector index
        self._verify_version(driver)
        self.embeddings = embeddings

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
        self, driver: Driver, query: str, params: Dict = {}
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
        with driver.session() as session:
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

        index_query = (
            "CALL db.index.vector.createNodeIndex("
            "$name,"
            "$label,"
            "$property,"
            "toInteger($dimensions),"
            "$similarity_fn )"
        )
        self.database_query(driver, index_query, params=index_data.dict())

    def drop_index(self, driver, name: str) -> None:
        """
        This method constructs a Cypher query and executes it
        to drop a vector index in Neo4j.
        """
        index_query = "DROP INDEX $name"
        parameters = {
            "name": name,
        }
        self.database_query(driver, index_query, params=parameters)

    def similarity_search(
        self,
        driver: Driver,
        name: str,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Performs the similarity search
        """
        try:
            if query_vector:
                validated_data = SimilaritySearchModel(
                    index_name=name, top_k=top_k, vector=query_vector
                )
            elif query_text:
                if not self.embeddings:
                    raise ValueError("Embedding method required for text query.")
                vector_embedding = self.embeddings.embed_query(query_text)
                validated_data = SimilaritySearchModel(
                    index_name=name, top_k=top_k, vector=vector_embedding
                )
            else:
                raise ValueError("Either query_vector or query_text must be provided.")

            parameters = validated_data.dict(exclude_none=True)

        except ValidationError as e:
            error_details = e.errors()
            raise ValueError(f"Validation failed: {error_details}")

        db_query_string = "CALL db.index.vector.queryNodes($index_name, $top_k, $vector) YIELD node, score, node.id AS id"
        return self.database_query(driver, db_query_string, params=parameters)
