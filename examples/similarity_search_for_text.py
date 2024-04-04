from typing import List
from neo4j import GraphDatabase
from neo4j_genai import VectorRetriever

from random import random
from neo4j_genai.embedder import Embedder
from neo4j_genai.indexes import create_vector_index

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

INDEX_NAME = "embedding-name"
DIMENSION = 1536

# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)


# Create Embedder object
class CustomEmbedder(Embedder):
    def embed_query(self, text: str) -> List[float]:
        return [random() for _ in range(DIMENSION)]


embedder = CustomEmbedder()

# Initialize the retriever
retriever = VectorRetriever(driver, embedder)

# Creating the index
create_vector_index(
    driver,
    INDEX_NAME,
    label="Document",
    property="propertyKey",
    dimensions=DIMENSION,
    similarity_fn="euclidean",
)

# Upsert the query
vector = [random() for _ in range(DIMENSION)]
insert_query = (
    "MERGE (n:Document)"
    "WITH n "
    "CALL db.create.setNodeVectorProperty(n, 'propertyKey', $vector)"
    "RETURN n"
)
parameters = {
    "vector": vector,
}
driver.execute_query(insert_query, parameters)

# Perform the similarity search for a text query
query_text = "hello world"
print(retriever.search(INDEX_NAME, query_text=query_text, top_k=5))
