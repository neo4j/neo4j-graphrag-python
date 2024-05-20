from langchain_openai import OpenAI
from neo4j import GraphDatabase
from neo4j_genai.retrievers.text_to_cypher import TextToCypherRetriever

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)

# Create LLM object
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

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

# Initialize the retriever
retriever = TextToCypherRetriever(driver=driver, llm=llm, neo4j_schema=neo4j_schema)

# (Optional) Provide user input/query pairs for the LLM to use as examples
examples = [
    "USER INPUT: 'Which actors starred in the Matrix?' QUERY: MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE m.title = 'The Matrix' RETURN p.name"
]

# Generate a Cypher query using the LLM, send it to the Neo4j database, and return the results
query_text = "Which movies did Hugo Weaving star in?"
print(retriever.search(query_text=query_text, examples=examples))
