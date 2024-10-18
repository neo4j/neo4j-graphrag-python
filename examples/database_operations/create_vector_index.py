import neo4j
from neo4j_graphrag.indexes import create_vector_index

NEO4J_URL = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")
INDEX_NAME = "vector_index"
DIMENSION = 1536

driver = neo4j.GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH)

create_vector_index(
    driver,
    INDEX_NAME,
    label="Document",
    embedding_property="vectorProperty",
    dimensions=DIMENSION,
    similarity_fn="euclidean",
)
