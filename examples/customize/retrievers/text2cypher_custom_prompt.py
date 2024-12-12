"""The example shows how to provide a custom prompt to Text2CypherRetriever.

Example using the OpenAILLM, hence the OPENAI_API_KEY needs to be set in the
environment for this example to run.
"""

import neo4j
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.schema import get_schema

# Define database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"

# Create LLM object
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

# (Optional) Specify your own Neo4j schema
# (also see get_structured_schema and get_schema functions)
neo4j_schema = """
Node properties:
User {name: STRING}
Person {name: STRING, born: INTEGER}
Movie {tagline: STRING, title: STRING, released: INTEGER}
Relationship properties:
ACTED_IN {roles: LIST}
DIRECTED {}
REVIEWED {summary: STRING, rating: INTEGER}
The relationships:
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:DIRECTED]->(:Movie)
(:User)-[:REVIEWED]->(:Movie)
"""

prompt = """Task: Generate a Cypher statement for querying a Neo4j graph database from a user input.

Do not use any properties or relationships not included in the schema.
Do not include triple backticks ``` or any additional text except the generated Cypher statement in your response.

Always filter movies that have not already been reviewed by the user with name: '{user_name}' using for instance:
(m:Movie)<-[:REVIEWED]-(:User {{name: <the_user_name>}})

Schema:
{schema}

Input:
{query_text}

Cypher query:
"""

with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
    # Initialize the retriever
    retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=neo4j_schema,
        # here we provide a custom prompt
        custom_prompt=prompt,
        neo4j_database=DATABASE,
    )

    # Generate a Cypher query using the LLM, send it to the Neo4j database, and return the results
    query_text = "Which movies did Hugo Weaving star in?"
    print(
        retriever.search(
            query_text=query_text,
            prompt_params={
                # you have to specify all placeholder except the {query_text} one
                "schema": get_schema(driver),
                "user_name": "the user asking question",
            },
        )
    )
