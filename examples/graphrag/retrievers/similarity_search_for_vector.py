"""This example assumes a Neo4j database is running and
contain some data with a vector index. Vector index name update
is required (see INDEX_NAME).

It shows how to use a vector-only retriever to find context
similar to a query **vector** using vector similarity.
"""
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever


INDEX_NAME = "my-index-name"  # UPDATE THIS LINE

# Connect to Neo4j database
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")
driver = GraphDatabase.driver(URI, auth=AUTH)

# Initialize the retriever
retriever = VectorRetriever(
    driver=driver,
    index_name=INDEX_NAME,
)

# Perform the similarity search for a vector query
query_vector = []
print(retriever.search(query_vector=query_vector, top_k=5))
