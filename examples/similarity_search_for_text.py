from typing import List
from neo4j import GraphDatabase
from neo4j_genai.client import GenAIClient

from random import random
from neo4j_genai.embeddings import Embeddings

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

INDEX_NAME = "embedding-name"
DIMENSION = 1536

# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)


# Create Embeddings object
class CustomEmbeddings(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        return [random() for _ in range(DIMENSION)]


embeddings = CustomEmbeddings()

# Initialize the client
client = GenAIClient(driver, embeddings)

client.drop_index(INDEX_NAME)

# Creating the index
client.create_index(
    INDEX_NAME,
    label="label",
    property="propertyKey",
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
client.database_query(insert_query, params=parameters)

# Perform the similarity search for a text query
query_text = "hello world"
print(client.similarity_search(INDEX_NAME, query_text=query_text, top_k=5))
