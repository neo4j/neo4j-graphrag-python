from typing import List
from neo4j import GraphDatabase
from src.client import GenAIClient

from random import random
from src.embeddings import Embeddings

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

INDEX_NAME = "embedding-name"
DIMENSION = 1536
driver = GraphDatabase.driver(URI, auth=AUTH)

# Create Embeddings object
class CustomEmbeddings(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        return [random() for _ in range(1536)]

embeddings = CustomEmbeddings()

# Initialize the client
client = GenAIClient(driver, embeddings)

client.drop_index(driver, INDEX_NAME)

# Creating the index
client.create_index(
    driver,
    INDEX_NAME,
    label="label",
    property="property",
    dimensions=DIMENSION,
    similarity_fn="euclidean",
)

# Upsert the query
vector = [random() for _ in range(DIMENSION)]
insert_query = (
    "MATCH (n:Node {id: $id})"
    "CALL db.create.setNodeVectorProperty(n, 'propertyKey', $vector)"
    "RETURN n"
)
parameters = {
    "id": 1,
    "vector": vector,
}
client.database_query(driver, insert_query, params=parameters)

# Perform the similarity search for a text query
query_text = "hello world"
print(client.similarity_search(
    driver, INDEX_NAME, query_text=query_text, top_k=5
))
