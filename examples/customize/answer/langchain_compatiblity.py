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
)

embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

retriever = VectorCypherRetriever(
    driver,
    index_name=INDEX,
    retrieval_query="WITH node, score RETURN node.title as title, node.plot as plot",
    embedder=embedder,  # type: ignore[arg-type, unused-ignore]
    neo4j_database=DATABASE,
)

# gpt-4.1 rather than gpt-5: LangChain's ChatOpenAI always sends a temperature
# (it defaults to 0.7), and gpt-5 only accepts the default (1) - so a gpt-5 model
# fails here even if this argument is omitted. Setting it explicitly also keeps
# the example working regardless of what ChatOpenAI defaults to.
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

rag = GraphRAG(
    retriever=retriever,
    llm=llm,  # type: ignore[arg-type, unused-ignore]
)

result = rag.search(
    "Tell me more about Avatar movies",
    return_context=False,
)
print(result.answer)

driver.close()
