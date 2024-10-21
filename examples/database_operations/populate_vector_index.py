import neo4j
from neo4j_graphrag.indexes import upsert_vector

NEO4J_URL = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")

driver = neo4j.GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH)

id = 1
embedding_property = "embedding"
vector = [1.0, 2.0, 3.0]

upsert_vector(driver, id, embedding_property, vector)
