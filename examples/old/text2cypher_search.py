from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers.text2cypher import Text2CypherRetriever

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)

# Create LLM object
llm = OpenAILLM(model_name="gpt-3.5-turbo", model_params={"temperature": 0})

# (Optional) Specify your own Neo4j schema
neo4j_schema = """
Node properties:
Person {name: STRING, born: INTEGER}
Movie {tagline: STRING, title: STRING, released: INTEGER}
Relationship properties:
ACTED_IN {roles: LIST}
REVIEWED {summary: STRING, rating: INTEGER}
The relationships:
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:DIRECTED]->(:Movie)
(:Person)-[:PRODUCED]->(:Movie)
(:Person)-[:WROTE]->(:Movie)
(:Person)-[:FOLLOWS]->(:Person)
(:Person)-[:REVIEWED]->(:Movie)
"""

# (Optional) Provide user input/query pairs for the LLM to use as examples
examples = [
    "USER INPUT: 'Which actors starred in the Matrix?' QUERY: MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE m.title = 'The Matrix' RETURN p.name"
]

# Initialize the retriever
retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    neo4j_schema=neo4j_schema,
    examples=examples,
)

# Generate a Cypher query using the LLM, send it to the Neo4j database, and return the results
query_text = "Which movies did Hugo Weaving star in?"
print(retriever.search(query_text=query_text))
