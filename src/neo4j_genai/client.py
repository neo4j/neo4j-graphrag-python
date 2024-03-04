from typing import List, Dict, Any, Optional
from pydantic import ValidationError
from neo4j import Driver
from neo4j.exceptions import CypherSyntaxError
from neo4j_genai.embeddings import Embeddings
from neo4j_genai.types import CreateIndexModel, SimilaritySearchModel, Neo4jRecord


class GenAIClient:
    """
    Provides functionality to use Neo4j's GenAI features
    """

    def __init__(
        self,
        driver: Driver,
        embeddings: Optional[Embeddings] = None,
    ) -> None:
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

        See Cypher manual on [Create node index](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_createNodeIndex)

        Args:
            name (str): The unique name of the index.
            label (str): The node label to be indexed.
            property (str): The property key of a node which contains embedding values.
            dimensions (int): Vector embedding dimension
            similarity_fn (str): case-insensitive values for the vector similarity function:
                ``euclidean`` or ``cosine``.

        Raises:
            ValueError: If validation of the input arguments fail.
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
        See Cypher manual on [Drop vector indexes](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-drop)

        Args:
            name (str): The name of the index to delete.
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
    ) -> List[Neo4jRecord]:
        """Get the top_k nearest neighbor embeddings for either provided query_vector or query_text.
        See the following documentation for more details:

        - [Query a vector index](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-query)
        - [db.index.vector.queryNodes()](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_queryNodes)

        Args:
            name (str): Refers to the unique name of the vector index to query.
            query_vector (Optional[List[float]], optional): The vector embeddings to get the closest neighbors of. Defaults to None.
            query_text (Optional[str], optional): The text to get the closest neighbors of. Defaults to None.
            top_k (int, optional): The number of neighbors to return. Defaults to 5.

        Raises:
            ValueError: If validation of the input arguments fail.
            ValueError: If no embeddings is provided.

        Returns:
            List[Neo4jRecord]: The `top_k` neighbors found in vector search with their nodes and scores.
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

        parameters = validated_data.dict(exclude_none=True)

        if query_text:
            if not self.embeddings:
                raise ValueError("Embedding method required for text query.")
            query_vector = self.embeddings.embed_query(query_text)
            parameters["query_vector"] = query_vector

        db_query_string = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector) 
        YIELD node, score
        """
        records = self.database_query(db_query_string, params=parameters)

        return [Neo4jRecord(node=record.node, score=record.score) for record in records]
