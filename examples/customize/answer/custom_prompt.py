"""End to end example of building a RAG pipeline backed by a Neo4j database.
Requires OPENAI_API_KEY to be in the env var.

This example illustrates:
- VectorCypherRetriever with a custom formatter function to extract relevant
    context from neo4j result
- Use of a custom prompt for RAG
- Logging configuration
"""

import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG, RagTemplate
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
    {query_text}

    Answer:
    """
)

rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=template)

result = rag.search("Tell me more about Avatar movies")
print(result.answer)

driver.close()
