"""End to end example of building a RAG pipeline backed by a Neo4j database.
Requires MISTRAL_API_KEY to be in the env var.

This example illustrates:
- VectorCypherRetriever with a custom formatter function to extract relevant
    context from neo4j result
- Logging configuration
"""

import logging

import neo4j
from neo4j_graphrag.embeddings.mistral import MistralAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm.mistralai_llm import MistralAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")
DATABASE = "neo4j"
INDEX_NAME = "moviePlotsEmbedding"


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

create_vector_index(
    driver,
    INDEX_NAME,
    label="Document",
    embedding_property="vectorProperty",
    dimensions=1024,
    similarity_fn="cosine",
)


embedder = MistralAIEmbeddings()

retriever = VectorCypherRetriever(
    driver,
    index_name=INDEX_NAME,
    retrieval_query="with node, score return node.title as title, node.plot as plot",
    result_formatter=formatter,
    embedder=embedder,
)

llm = MistralAILLM(model_name="mistral-small-latest")

rag = GraphRAG(retriever=retriever, llm=llm)

result = rag.search("Tell me more about Avatar movies")
print(result.answer)

driver.close()
