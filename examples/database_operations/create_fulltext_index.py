import neo4j
from neo4j_graphrag.indexes import create_fulltext_index

NEO4J_URL = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")
FULLTEXT_INDEX_NAME = "fulltext_index"

driver = neo4j.GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH)

create_fulltext_index(
    driver, FULLTEXT_INDEX_NAME, label="Document", node_properties=["textProperty"]
)
