from neo4j import GraphDatabase
from neo4j_genai.retrievers import PineconeNeo4jRetriever
from pinecone import Pinecone

from examples.embedding_biology import EMBEDDING_BIOLOGY

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
