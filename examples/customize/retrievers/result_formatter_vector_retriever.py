"""This example demonstrates how to customize the retriever
results format.

Retriever.get_search_result returns a RawSearchResult object that consists
in a list of neo4j.Records and an optional metadata dictionary. The
Retriever.search method returns a RetrieverResult object where each neo4j.Record
has been replaced by a RetrieverResultItem, ie a content and metadata dictionary.
The content is what will be used to augment the prompt. By default, this content
is a stringified representation of the neo4j.Record. There are multiple ways the
user can act on this format:
- Use the `return_properties` parameter
- And/or use the result_formatter parameter

Let's consider the movie database where the movies' plot have been embedded. Movie
nodes have additional properties such as title and are connected to Actor nodes that
have a name property:

(:Movie {embedding: [], title: "", plot: "", year: "", budget: 1000, ....})
    <-[:ACTED_IN]-
    (:Actor {name: ...})

NB: to run this example OPENAI_API_KEY needs to be in the env vars.
To use another embedder, see the corresponding examples in ../customize/embeddings
"""

import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.types import RetrieverResultItem

URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX_NAME = "moviePlotsEmbedding"


# Connect to Neo4j database
driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)


query_text = "Find a movie about astronauts"
top_k_results = 1


"""First, let's select the properties we want to return
with the return_properties parameter:
"""
print("=" * 50)
print("RETURN PROPERTIES")
retriever = VectorRetriever(
    driver=driver,
    index_name=INDEX_NAME,
    embedder=OpenAIEmbeddings(),
    return_properties=["title", "plot"],
    neo4j_database=DATABASE,
)
print(retriever.search(query_text=query_text, top_k=top_k_results))
print()
"""
OUTPUT:
RetrieverResult(
    items=[
        RetrieverResultItem(
            content="{'title': 'For All Mankind', 'plot': 'This movie documents the Apollo missions perhaps the most definitively of any movie under two hours. Al Reinert watched all the footage shot during the missions--over 6,000,000 feet of it, ...'}",
            metadata={'score': 0.9354040622711182, 'nodeLabels': None, 'id': None}
        )
    ]
    metadata={'__retriever': 'VectorRetriever'}
)
"""


"""In a second example, we'll use the ability to format the result in detail
with a function
"""
print("=" * 50)
print("RESULT FOMATTER")


def my_result_formatter(record: neo4j.Record) -> RetrieverResultItem:
    """
    If 'return_properties' are not specified, vector retrievers will return a record with keys:
    - `node`: a dict representation of all node properties (except embedding if we can identify it)
    - `id`: the node element ID
    - `nodeLabels`: the labels attached to the node
    - `score`: the score returned by the vector index search that tells us how close the node vector is from the query vector

    In the case of movies, we may want to keep in the content only the title and plot
    (passed to the LLM afterward) and keep in the metadata the score.
    This can be achieved with this function:
    """
    node = record.get("node")
    return RetrieverResultItem(
        content=f"{node.get('title')}: {node.get('plot')}",
        metadata={"score": record.get("score")},
    )


retriever = VectorRetriever(
    driver=driver,
    index_name=INDEX_NAME,
    embedder=OpenAIEmbeddings(),
    result_formatter=my_result_formatter,
)

query_text = "Find a movie about astronauts"
print(retriever.search(query_text=query_text, top_k=top_k_results))
print()
"""
OUTPUT:
RetrieverResult(
    items=[
        RetrieverResultItem(
            content='For All Mankind: This movie documents the Apollo missions perhaps the most definitively of any movie under two hours. Al Reinert watched all the footage shot during the missions--over 6,000,000 feet of it, ...',
            metadata={'score': 0.9354040622711182}
        )
    ]
    metadata={'__retriever': 'VectorRetriever'}
)
"""


"""We can mix both return_properties and result_formatter:
"""
print("=" * 50)
print("RETURN PROPERTIES + RESULT FOMATTER")

retriever = VectorRetriever(
    driver=driver,
    index_name=INDEX_NAME,
    embedder=OpenAIEmbeddings(),
    return_properties=["title", "plot"],
    result_formatter=my_result_formatter,
)

query_text = "Find a movie about astronauts"
print(retriever.search(query_text=query_text, top_k=top_k_results))
print()

"""
OUTPUT:
RetrieverResult(
    items=[
        RetrieverResultItem(
            content='For All Mankind: This movie documents the Apollo missions perhaps the most definitively of any movie under two hours. Al Reinert watched all the footage shot during the missions--over 6,000,000 feet of it, ...',
            metadata={'score': 0.9354040622711182})
    ]
    metadata={'__retriever': 'VectorRetriever'}
)
"""

driver.close()
