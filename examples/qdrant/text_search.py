from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from qdrant_client import QdrantClient

NEO4J_URL = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")


def main() -> None:
    with GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH) as neo4j_driver:
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        retriever = QdrantNeo4jRetriever(
            driver=neo4j_driver,
            client=QdrantClient(url="http://localhost:6333"),
            collection_name="Jeopardy",
            id_property_external="neo4j_id",
            id_property_neo4j="id",
            embedder=embedder,  # type: ignore
        )

        res = retriever.search(query_text="biology", top_k=2)
        print(res)


if __name__ == "__main__":
    main()
