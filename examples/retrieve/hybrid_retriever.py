"""This example uses an example Movie database where movies' plots are embedded
using OpenAI embeddings. OPENAI_API_KEY needs to be set in the environment for
this example to run.

It shows how to use a hybrid retriever to find context
similar to a query **text** using vector+text similarity.
"""

import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import HybridRetriever

# Define database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX_NAME = "moviePlotsEmbedding"
FULLTEXT_INDEX_NAME = "movieFulltext"


with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
    # Initialize the retriever
    retriever = HybridRetriever(
        driver=driver,
        vector_index_name=INDEX_NAME,
        fulltext_index_name=FULLTEXT_INDEX_NAME,
        embedder=OpenAIEmbeddings(),
        # optionally, provide a list of properties to fetch (default fetch all)
        # return_properties=[],
        # optionally, configure how to format the results
        # (see corresponding example in 'customize' directory)
        # result_formatter=None,
        # optionally, set neo4j database
        neo4j_database=DATABASE,
    )

    # Perform the similarity search for a text query
    # (retrieve the top 5 most similar nodes)
    query_text = "Find me a movie about aliens"
    results = retriever.search(
        query_text=query_text,
        top_k=5,
        threshold_vector_index=0.1,
        threshold_fulltext_index=0.8,
    )

    print(results.items[0].metadata)

    # note: it is also possible to query from a query_vector directly:
    # query_vector: list[float] = [...]
    # retriever.search(query_vector=query_vector, top_k=5)
