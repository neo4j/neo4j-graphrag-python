from neo4j import GraphDatabase
from neo4j_genai.retrievers import VectorRetriever

from random import random
from neo4j_genai.indexes import create_vector_index

from langchain_openai import OpenAIEmbeddings

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

INDEX_NAME = "embedding-name-large"
DIMENSION = 3072

# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)


# Create Embedder object
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize the retriever
retriever = VectorRetriever(driver, INDEX_NAME, embedder)  # type: ignore

# Creating the index
create_vector_index(
    driver,
    INDEX_NAME,
    label="Document",
    property="propertyKey",
    dimensions=DIMENSION,
    similarity_fn="cosine",
)

# Upsert the query
vector = [random() for _ in range(DIMENSION)]

insert_query = (
    "MERGE (n:Document {id: $id})"
    "WITH n "
    "CALL db.create.setNodeVectorProperty(n, 'propertyKey', $vector)"
    "RETURN n"
)
parameters = {
    "id": 0,
    "vector": vector,
}
driver.execute_query(insert_query, parameters)

# Perform the similarity search for a text query
query_text = "Find me a book about Fremen"
print(retriever.search(query_text=query_text, top_k=5))
