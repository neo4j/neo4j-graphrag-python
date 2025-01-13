"""End to end example of building a RAG pipeline backed by a Neo4j database.
Requires OPENAI_API_KEY to be in the env var.

This example illustrates:
- VectorCypherRetriever with a custom formatter function to extract relevant
    context from neo4j result
- Logging configuration
"""

import logging

import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

# Define database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX = "moviePlotsEmbedding"


# setup logger config
logger = logging.getLogger("neo4j_graphrag")
logging.basicConfig(format="%(asctime)s - %(message)s")
logger.setLevel(logging.DEBUG)


def formatter(record: neo4j.Record) -> RetrieverResultItem:
    return RetrieverResultItem(content=f'{record.get("title")}: {record.get("plot")}')


driver = neo4j.GraphDatabase.driver(
    URI,
    auth=AUTH,
)

embedder = OpenAIEmbeddings()

retriever = VectorCypherRetriever(
    driver,
    index_name=INDEX,
    retrieval_query="with node, score return node.title as title, node.plot as plot",
    result_formatter=formatter,
    embedder=embedder,
    neo4j_database=DATABASE,
)

llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

rag = GraphRAG(retriever=retriever, llm=llm)

result = rag.search(
    "Tell me more about Avatar movies",
    return_context=True,
)
print(result.answer)
# print(result.retriever_result)

driver.close()
