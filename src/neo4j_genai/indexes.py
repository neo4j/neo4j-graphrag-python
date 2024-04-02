from typing import List

from neo4j import Driver
from pydantic import ValidationError
from .types import CreateIndexModel


def create_vector_index(
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

    See Cypher manual on [Create node index](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_createNodeIndex)

    Args:
        driver (Driver): Neo4j Python driver instance.
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
        CreateIndexModel(
            **{
                "name": name,
                "label": label,
                "property": property,
                "dimensions": dimensions,
                "similarity_fn": similarity_fn,
            }
        )
    except ValidationError as e:
        raise ValueError(f"Error for inputs to create_index {str(e)}")

    query = (
        f"CREATE VECTOR INDEX $name IF NOT EXISTS FOR (n:{label}) ON n.{property} OPTIONS "
        "{ indexConfig: { `vector.dimensions`: toInteger($dimensions), `vector.similarity_function`: $similarity_fn } }"
    )
    driver.execute_query(
        query, {"name": name, "dimensions": dimensions, "similarity_fn": similarity_fn}
    )


def create_fulltext_index(
    driver: Driver, name: str, label: str, text_node_properties: List[str] = []
) -> None:
    """ """
    query = (
        "CREATE FULLTEXT INDEX $name"
        f"FOR (n:`{label}`) ON EACH "
        f"[{', '.join(['n.`' + property + '`' for property in text_node_properties])}]"
    )
    driver.execute_query(query, {"name": name})


def drop_vector_index(driver: Driver, name: str) -> None:
    """
    This method constructs a Cypher query and executes it
    to drop a vector index in Neo4j.
    See Cypher manual on [Drop vector indexes](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-drop)

    Args:
        driver (Driver): Neo4j Python driver instance.
        name (str): The name of the index to delete.
    """
    query = "DROP INDEX $name"
    parameters = {
        "name": name,
    }
    driver.execute_query(query, parameters)
