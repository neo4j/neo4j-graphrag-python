"""This example uses an example Movie database where movies' plots are embedded
using OpenAI embeddings. OPENAI_API_KEY needs to be set in the environment for
this example to run.

It shows how to use a vector-only retriever to find context
similar to a query **text** using vector similarity.
"""

import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

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
        embedder=OpenAIEmbeddings(),
        # optionally, provide a list of properties to fetch (default fetch all)
        # return_properties=[],
        # optionally, configure how to format the results
        # (see corresponding example in 'customize' directory)
        # result_formatter=None,
        # optionally, set neo4j database
        # neo4j_database="neo4j",
    )

    # Perform the similarity search for a text query
    # (retrieve the top 5 most similar nodes)
    query_text = "Find me a movie about aliens"
    print(retriever.search(query_text=query_text, top_k=5))
