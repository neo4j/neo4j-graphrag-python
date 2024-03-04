from neo4j import GraphDatabase
from neo4j_genai.client import GenAIClient

from random import random

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

INDEX_NAME = "embedding-name"
DIMENSION = 1536

# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)

# Initialize the client
client = GenAIClient(driver)

try:
    client.drop_index(INDEX_NAME)
except DatabaseError as e:
    print(e)

# Creating the index
client.create_index(
    INDEX_NAME,
    label="Document",
    property="propertyKey",
    dimensions=DIMENSION,
    similarity_fn="euclidean",
)

# Upsert the vector
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
client.database_query(insert_query, params=parameters)

# Perform the similarity search for a vector query
query_vector = [random() for _ in range(DIMENSION)]
print(client.similarity_search(INDEX_NAME, query_vector=query_vector, top_k=5))
