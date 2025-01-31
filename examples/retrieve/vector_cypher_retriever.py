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
        collect { MATCH (actor:Actor)-[:ACTED_IN]->(node) RETURN actor.name } AS actors,
        score as similarityScore
"""

with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
    # Initialize the retriever
    retriever = VectorCypherRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        # note: embedder is optional if you only use query_vector
        embedder=OpenAIEmbeddings(),
        retrieval_query=RETRIEVAL_QUERY,
        # optionally, configure how to format the results
        # (see corresponding example in 'customize' directory)
        # result_formatter=None,
        # optionally, set neo4j database
        neo4j_database=DATABASE,
    )

    # Perform the similarity search for a text query
    # (retrieve the top 5 most similar nodes)
    query_text = "Who were the actors in Avatar?"
    print(retriever.search(query_text=query_text, top_k=5))

    # note: it is also possible to query from a query_vector directly:
    # query_vector: list[float] = [...]
    # retriever.search(query_vector=query_vector, top_k=5)
