"""End to end example of building a RAG pipeline backed by a Neo4j database.
Requires OPENAI_API_KEY to be in the env var.

This example illustrates:
- VectorCypherRetriever with a custom formatter function to extract relevant
    context from neo4j result
- Use of a custom prompt for RAG
- Logging configuration
"""

import logging
import neo4j

from neo4j_genai.types import RetrieverResultItem
from neo4j_genai.embeddings.openai import OpenAIEmbeddings
from neo4j_genai import VectorCypherRetriever, GraphRAG, OpenAILLM, RagTemplate

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")
DATABASE = "neo4j"
INDEX = "moviePlotsEmbedding"


# setup logger config
logger = logging.getLogger("neo4j_genai")
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
    embedder=embedder,
)

llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

template = RagTemplate(
    template="""You are an expert at movies and actors. Your task is to
    answer the user's question based on the provided context. Use only the
    information within that context.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
)

rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=template)

result = rag.search("Tell me more about Avatar movies")
print(result.content)

driver.close()
