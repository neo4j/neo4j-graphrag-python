"""This example demonstrates how to use PineconeNeo4jRetriever, ie vectors are
stored in the Pinecone database.

See the [README](./README.md) for more
information about how spin up a Pinecone and Neo4j databases if needed.

In this example, search is performed from an already computed vector.
"""

from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import PineconeNeo4jRetriever
from pinecone import Pinecone

from examples.data.embedding_biology import EMBEDDING_BIOLOGY

NEO4J_AUTH = ("neo4j", "password")
NEO4J_URL = "neo4j://localhost:7687"
PC_API_KEY = "API_KEY"


def main() -> None:
    with GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH) as neo4j_driver:
        pc_client = Pinecone(PC_API_KEY)
        retriever = PineconeNeo4jRetriever(
            driver=neo4j_driver,
            client=pc_client,
            index_name="jeopardy",
            id_property_neo4j="id",
        )

        res = retriever.search(query_vector=EMBEDDING_BIOLOGY, top_k=2)
        print(res)


if __name__ == "__main__":
    main()
