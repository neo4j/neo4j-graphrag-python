"""This example uses an example Movie database where movies' plots are embedded
using OpenAI embeddings. OPENAI_API_KEY needs to be set in the environment for
this example to run.

It shows how to use a vector-only retriever to find context
similar to a query **vector** using vector similarity.
"""

import neo4j
from neo4j_graphrag.retrievers import VectorRetriever

from examples.data.embeddings_avatar import EMBEDDINGS_AVATAR

# Define database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX_NAME = "moviePlotsEmbedding"


with neo4j.GraphDatabase.driver(URI, auth=AUTH, database=DATABASE) as driver:
    # Initialize the retriever
    retriever = VectorRetriever(
        driver=driver,
        index_name=INDEX_NAME,
    )

    # Perform the similarity search for a vector query
    query_vector: list[float] = EMBEDDINGS_AVATAR
    print(retriever.search(query_vector=query_vector, top_k=5))
