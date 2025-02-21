"""End to end example of building a RAG pipeline backed by a Neo4j database,
simulating a chat with message history which is also stored in Neo4j.

Requires OPENAI_API_KEY to be in the env var.
"""

import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.message_history import Neo4jMessageHistory
from neo4j_graphrag.retrievers import VectorCypherRetriever

# Define database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX = "moviePlotsEmbedding"


driver = neo4j.GraphDatabase.driver(
    URI,
    auth=AUTH,
)

embedder = OpenAIEmbeddings()

retriever = VectorCypherRetriever(
    driver,
    index_name=INDEX,
    retrieval_query="""
        WITH node as movie, score
        CALL(movie) {
            MATCH (movie)<-[:ACTED_IN]-(p:Person)
            RETURN collect(p.name) as actors
        }
        CALL(movie) {
            MATCH (movie)<-[:DIRECTED]-(p:Person)
            RETURN collect(p.name) as directors
        }
        RETURN movie.title as title, movie.plot as plot, movie.year as year, actors, directors
    """,
    embedder=embedder,
    neo4j_database=DATABASE,
)

llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

rag = GraphRAG(
    retriever=retriever,
    llm=llm,
)

history = Neo4jMessageHistory(session_id="123", driver=driver, window=10)

questions = [
    "Who starred in the Apollo 13 movies?",
    "Who was its director?",
    "In which year was this movie released?",
]

for question in questions:
    result = rag.search(
        question,
        return_context=False,
        message_history=history,
    )

    answer = result.answer
    print("#" * 50, question)
    print(answer)
    print("#" * 50)

    history.add_message(
        {
            "role": "user",
            "content": question,
        }
    )
    history.add_message(
        {
            "role": "assistant",
            "content": answer,
        }
    )

driver.close()
