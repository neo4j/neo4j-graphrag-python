from __future__ import annotations

import neo4j
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX_NAME = "moviePlotsEmbedding"
DIMENSION = 1536


# Connect to Neo4j database
with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
    # Initialize the retriever
    retriever = VectorRetriever(driver, INDEX_NAME, embedder=OpenAIEmbeddings())

    # Perform the search
    query_text = "Find me a movie about love"
    pre_filters = {"int_property": {"$gt": 100}}
    # pre_filters = {
    #     "year": {
    #         "$nin": ["1999", "2000"]
    #     }
    # }
    retriever_result = retriever.search(
        query_text=query_text,
        top_k=1,
        filters=pre_filters,
    )
    print(retriever_result)
