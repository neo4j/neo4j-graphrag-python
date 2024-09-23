"""End to end example of building a RAG pipeline backed by a Neo4j database.
Requires OPENAI_API_KEY to be in the env var.

This example illustrates:
- VectorCypherRetriever with a custom formatter function to extract relevant
    context from neo4j result
- Logging configuration
"""

import logging

import neo4j
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")
DATABASE = "neo4j"
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
    database=DATABASE,
)

embedder = OpenAIEmbeddings()

retriever = VectorCypherRetriever(
    driver,
    index_name=INDEX,
    retrieval_query="with node, score return node.title as title, node.plot as plot",
    format_record_function=formatter,
    embedder=embedder,  # type: ignore
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)  # type: ignore

rag = GraphRAG(retriever=retriever, llm=llm)

result = rag.search("Tell me more about Avatar movies")
print(result.answer)

driver.close()
