"""This example uses an example Movie database where movies' plots are embedded
using OpenAI embeddings. OPENAI_API_KEY needs to be set in the environment for
this example to run.

Also requires minimal Cypher knowledge to write the retrieval query.

It shows how to use a vector-cypher retriever to find context
similar to a query **text** using vector similarity + graph traversal.
"""

import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

# Define database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX_NAME = "moviePlotsEmbedding"

# for each Movie node matched by the vector search, retrieve more context:
# the name of all actors starring in that movie
RETRIEVAL_QUERY = """
RETURN  node.title as movieTitle,
        node.plot as moviePlot,
        collect { MATCH (actor:Actor)-[:ACTED_IN]->(node) RETURN a.name } AS actors,
        score as similarityScore
"""


def my_result_formatter(record: neo4j.Record) -> RetrieverResultItem:
    """The record is a row output from the RETRIEVAL_QUERY so it our case it contains
    the following keys:
    - movieTitle
    - moviePlot
    - actors
    - similarityScore
    """
    return RetrieverResultItem(
        content=f"Movie title: {record.get('movieTitle')}, Plot: {record.get('moviePlot')}, Actors: {record.get('actors')}",
        metadata={"score": record.get("similarityScore")},
    )


with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
    # Initialize the retriever
    retriever = VectorCypherRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        # note: embedder is optional if you only use query_vector
        embedder=OpenAIEmbeddings(),
        retrieval_query=RETRIEVAL_QUERY,
        result_formatter=my_result_formatter,
        # optionally, set neo4j database
        neo4j_database=DATABASE,
    )

    # Perform the similarity search for a text query
    # (retrieve the top 5 most similar nodes)
    query_text = "Who were the actors in Avatar?"
    print(retriever.search(query_text=query_text, top_k=5))
