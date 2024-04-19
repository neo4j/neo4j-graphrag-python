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

from typing import List

from neo4j import Driver
from pydantic import ValidationError
from .types import VectorIndexModel, FulltextIndexModel


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

    See Cypher manual on [Create vector index](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/#indexes-vector-create)

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
        VectorIndexModel(
            **{
                "driver": driver,
                "name": name,
                "label": label,
                "property": property,
                "dimensions": dimensions,
                "similarity_fn": similarity_fn,
            }
        )
    except ValidationError as e:
        raise ValueError(f"Error for inputs to create_vector_index {str(e)}")

    query = (
        f"CREATE VECTOR INDEX $name FOR (n:{label}) ON n.{property} OPTIONS "
        "{ indexConfig: { `vector.dimensions`: toInteger($dimensions), `vector.similarity_function`: $similarity_fn } }"
    )
    driver.execute_query(
        query, {"name": name, "dimensions": dimensions, "similarity_fn": similarity_fn}
    )


def create_fulltext_index(
    driver: Driver, name: str, label: str, node_properties: List[str]
) -> None:
    """
    This method constructs a Cypher query and executes it
    to create a new fulltext index in Neo4j.

    See Cypher manual on [Create fulltext index](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/#create-full-text-indexes)

    Args:
        driver (Driver): Neo4j Python driver instance.
        name (str): The unique name of the index.
        label (str): The node label to be indexed.
        node_properties (List[str]): The node properties to create the fulltext index on.

    Raises:
        ValueError: If validation of the input arguments fail.
    """
    try:
        FulltextIndexModel(
            **{
                "driver": driver,
                "name": name,
                "label": label,
                "node_properties": node_properties,
            }
        )
    except ValidationError as e:
        raise ValueError(f"Error for inputs to create_fulltext_index {str(e)}")

    query = (
        "CREATE FULLTEXT INDEX $name"
        f"FOR (n:`{label}`) ON EACH "
        f"[{', '.join(['n.`' + prop + '`' for prop in node_properties])}]"
    )
    driver.execute_query(query, {"name": name})


def drop_index(driver: Driver, name: str) -> None:
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
