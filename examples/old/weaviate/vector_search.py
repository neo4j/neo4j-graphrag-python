from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import WeaviateNeo4jRetriever

from examples.old.embedding_biology import EMBEDDING_BIOLOGY
from weaviate.connect.helpers import connect_to_local

NEO4J_URL = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")


def main() -> None:
    with GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH) as neo4j_driver:
        with connect_to_local() as w_client:
            retriever = WeaviateNeo4jRetriever(
                driver=neo4j_driver,
                client=w_client,
                collection="Jeopardy",
                id_property_external="neo4j_id",
                id_property_neo4j="id",
            )
            res = retriever.search(query_vector=EMBEDDING_BIOLOGY, top_k=2)
            print(res)


if __name__ == "__main__":
    main()
