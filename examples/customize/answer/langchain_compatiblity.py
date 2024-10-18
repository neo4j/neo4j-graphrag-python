"""The LLM interface is compatible with LangChain chat API,
 so any LangChain implementation can be used. For instance,
 in GraphRAG:

Requires OPENAI_API_KEY to be in the env var.
"""

import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")
DATABASE = "neo4j"
INDEX = "moviePlotsEmbedding"


driver = neo4j.GraphDatabase.driver(
    URI,
    auth=AUTH,
    database=DATABASE,
)

embedder = OpenAIEmbeddings()

retriever = VectorCypherRetriever(
    driver,
    index_name=INDEX,
    retrieval_query="WITH node, score RETURN node.title as title, node.plot as plot",
    embedder=embedder,  # type: ignore
)

llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

rag = GraphRAG(retriever=retriever, llm=llm)

result = rag.search("Tell me more about Avatar movies")
print(result.answer)

driver.close()
