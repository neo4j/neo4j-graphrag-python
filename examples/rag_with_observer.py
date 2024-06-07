"""End to end example of building a RAG pipeline backed by a Neo4j database.
Requires OPENAI_API_KEY to be in the env var.

This example illustrates:
- Log observer
- Logging configuration
"""

import logging
import neo4j

from neo4j_genai.embeddings.openai import OpenAIEmbeddings
from neo4j_genai.types import RetrieverResultItem
from neo4j_genai import VectorCypherRetriever, RAG, OpenAILLM
from neo4j_genai.observers import LogObserver

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")
DATABASE = "neo4j"
INDEX = "moviePlotsEmbedding"


# setup logger config
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s")

internal_logger = logging.getLogger("neo4j_genai")
internal_logger.setLevel(logging.WARNING)

logger = logging.getLogger("my_logger")
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
    embedder=embedder,
)

llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

observer = LogObserver(
    logger=logger,
    level=logging.INFO,
)

rag = RAG(retriever=retriever, llm=llm, observers=[observer])

result = rag.search("Tell me more about Avatar movies")
print(result)

driver.close()
