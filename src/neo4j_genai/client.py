from typing import List, Optional
from pydantic import ValidationError
from neo4j import Driver
from .embedder import Embedder
from .types import CreateIndexModel, SimilaritySearchModel, Neo4jRecord


class GenAIClient:
    """
    Provides functionality to use Neo4j's GenAI features
    """

    def __init__(
        self,
        driver: Driver,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.driver = driver
        self._verify_version()
        self.embedder = embedder

    def _verify_version(self) -> None:
        """
        Check if the connected Neo4j database version supports vector indexing.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.18.1) that is known to support vector
        indexing. Raises a ValueError if the connected Neo4j version is
        not supported.
        """
        records, _, _ = self.driver.execute_query("CALL dbms.components()")
        version = records[0]["versions"][0]

        if "aura" in version:
            version_tuple = (
                *tuple(map(int, version.split("-")[0].split("."))),
                0,
            )
            target_version = (5, 18, 0)
        else:
            version_tuple = tuple(map(int, version.split(".")))
            target_version = (5, 18, 1)


        if version_tuple < target_version:
            raise ValueError(
                "This package only supports Neo4j version 5.18.1 or greater"
            )

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
        try:
            CreateIndexModel(**{
                "name": name,
                "label": label,
                "property": property,
                "dimensions": dimensions,
                "similarity_fn": similarity_fn,
        })
        except ValidationError as e:
            raise ValueError(f"Error for inputs to create_index {str(e)}")

        query = (
            f"CREATE VECTOR INDEX $name IF NOT EXISTS FOR (n:{label}) ON n.{property} OPTIONS "
            "{ indexConfig: { `vector.dimensions`: toInteger($dimensions), `vector.similarity_function`: $similarity_fn } }"
        )
        self.driver.execute_query(query, {"name": name, "dimensions": dimensions, "similarity_fn": similarity_fn})

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
        self.driver.execute_query(query, parameters)

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
            ValueError: If no embedder is provided.

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

        parameters = validated_data.model_dump(exclude_none=True)

        if query_text:
            if not self.embedder:
                raise ValueError("Embedding method required for text query.")
            query_vector = self.embedder.embed_query(query_text)
            parameters["query_vector"] = query_vector
            del parameters["query_text"]

        db_query_string = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        """
        records, _, _ = self.driver.execute_query(db_query_string, parameters)

        try:
            return [
                Neo4jRecord(node=record["node"], score=record["score"])
                for record in records
            ]
        except ValidationError as e:
            error_details = e.errors()
            raise ValueError(
                f"Validation failed while constructing output: {error_details}"
            )
