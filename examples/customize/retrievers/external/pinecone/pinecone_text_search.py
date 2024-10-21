"""This example demonstrates how to use PineconeNeo4jRetriever, ie vectors are
stored in the Pinecone database.

See the [README](./README.md) for more
information about how spin up a Pinecone and Neo4j databases if needed.

In this example, search is performed from a text. Embeddings are computed
using OpenAI models. See [../../embeddings/](../../embeddings/) for examples
using other supported embedders.
"""

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import PineconeNeo4jRetriever
from pinecone import Pinecone

NEO4J_AUTH = ("neo4j", "password")
NEO4J_URL = "neo4j://localhost:7687"
PC_API_KEY = "API_KEY"


def main() -> None:
    with GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH) as neo4j_driver:
        pc_client = Pinecone(PC_API_KEY)
        embedder = OpenAIEmbeddings()

        retriever = PineconeNeo4jRetriever(
            driver=neo4j_driver,
            client=pc_client,
            index_name="jeopardy",
            id_property_neo4j="id",
            embedder=embedder,
        )

        res = retriever.search(query_text="biology", top_k=2)
        print(res)


if __name__ == "__main__":
    main()
