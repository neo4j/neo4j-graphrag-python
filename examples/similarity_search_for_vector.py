from neo4j import GraphDatabase
from src.client import GenAIClient

from random import random

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

INDEX_NAME = "embedding-name"
DIMENSION = 1536

# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)

# Initialize the client
client = GenAIClient(driver)

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

# Upsert the vector
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

# Perform the similarity search for a vector query
query_vector = [random() for _ in range(DIMENSION)]
# retriever_query
client.similarity_search(driver, INDEX_NAME, query_vector=query_vector, top_k=5)

retriever_query = "WITH node, score MATCH (node)-->(parent) RETURN parent.text"
client.similarity_search(driver, retriever_query, INDEX_NAME, query_vector=query_vector, top_k=5)
