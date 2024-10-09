from __future__ import annotations

import random
import string

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

INDEX_NAME = "embedding-name"
DIMENSION = 1536

# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)


# Create Embedder object
class CustomEmbedder(Embedder):
    def embed_query(self, text: str) -> list[float]:
        return [random.random() for _ in range(DIMENSION)]


# Generate random strings
def random_str(n: int) -> str:
    return "".join([random.choice(string.ascii_letters) for _ in range(n)])


embedder = CustomEmbedder()

# Creating the index
create_vector_index(
    driver,
    INDEX_NAME,
    label="Document",
    embedding_property="vectorProperty",
    dimensions=DIMENSION,
    similarity_fn="euclidean",
)

# Initialize the retriever
retriever = VectorRetriever(driver, INDEX_NAME, embedder)

# Upsert the query
vector = [random.random() for _ in range(DIMENSION)]
insert_query = (
    "MERGE (doc:Document {id: $id})"
    "ON CREATE SET  doc.int_property = $id, "
    "               doc.short_text_property = toString($id)"
    "WITH doc "
    "CALL db.create.setNodeVectorProperty(doc, 'vectorProperty', $vector)"
    "WITH doc "
    "MERGE (author:Author {name: $authorName})"
    "MERGE (doc)-[:AUTHORED_BY]->(author)"
    "RETURN doc, author"
)
parameters = {
    "id": random.randint(0, 10000),
    "vector": vector,
    "authorName": random_str(10),
}
driver.execute_query(insert_query, parameters)

# Perform the search
query_text = "Find me a book about Fremen"
print(
    retriever.search(
        query_text=query_text, top_k=1, filters={"int_property": {"$gt": 100}}
    )
)
