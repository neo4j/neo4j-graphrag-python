"""The LLM interface is compatible with LangChain chat API,
 so any LangChain implementation can be used. Same for embedders.
 For instance, in GraphRAG:

Requires OPENAI_API_KEY to be in the env var.
"""

import neo4j
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import VectorCypherRetriever

# Define database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX = "moviePlotsEmbedding"


driver = neo4j.GraphDatabase.driver(
    URI,
    auth=AUTH,
    database=DATABASE,
)

embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

retriever = VectorCypherRetriever(
    driver,
    index_name=INDEX,
    retrieval_query="WITH node, score RETURN node.title as title, node.plot as plot",
    embedder=embedder,  # type: ignore
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

rag = GraphRAG(
    retriever=retriever,
    llm=llm,  # type: ignore
)

result = rag.search("Tell me more about Avatar movies")
print(result.answer)

driver.close()
