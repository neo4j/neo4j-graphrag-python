from __future__ import annotations

import neo4j
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

INDEX_NAME = "embedding-name"
DIMENSION = 1536

# Connect to Neo4j database
driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)


# Initialize the retriever
retriever = VectorRetriever(driver, INDEX_NAME, embedder=OpenAIEmbeddings())

# Perform the search
query_text = "Find me a book about Fremen"
print(
    retriever.search(
        query_text=query_text,
        top_k=1,
        filters={"int_property": {"$gt": 100}}
    )
)
