"""The example leverages the Text2CypherRetriever to fetch some context.
It uses the OpenAILLM, hence the OPENAI_API_KEY needs to be set in the
environment for this example to run.
"""

import neo4j
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever

# Define database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"

# Create LLM object
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

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

with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
    # Initialize the retriever
    retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=neo4j_schema,
        examples=examples,
        # optionally, you can also provide your own prompt
        # for the text2Cypher generation step
        # custom_prompt="",
        neo4j_database=DATABASE,
    )

    # Generate a Cypher query using the LLM, send it to the Neo4j database, and return the results
    query_text = "Which movies did Hugo Weaving star in?"
    print(retriever.search(query_text=query_text))
