from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from qdrant_client import QdrantClient

from examples.embedding_biology import EMBEDDING_BIOLOGY

NEO4J_URL = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")


def main() -> None:
    with GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH) as neo4j_driver:
        retriever = QdrantNeo4jRetriever(
            driver=neo4j_driver,
            client=QdrantClient(url="http://localhost:6333"),
            collection_name="Jeopardy",
            id_property_external="neo4j_id",
            id_property_neo4j="id",
        )
        res = retriever.search(query_vector=EMBEDDING_BIOLOGY, top_k=2)
        print(res)


if __name__ == "__main__":
    main()
