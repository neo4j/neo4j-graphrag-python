"""End to end example of building a RAG pipeline backed by a Neo4j database.
Requires OPENAI_API_KEY to be in the env var.

This example illustrates:
- VectorCypherRetriever with a custom formatter function to extract relevant
    context from neo4j result
- Logging configuration
- Function calling
"""

import logging
import json

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

# Function calling
function_defs = [
    {
        "name": "get_streaming_availability",
        "description": "Checks which streaming platforms a movie is currently available on.",
        "parameters": {
            "type": "object",
            "properties": {
                "movie_title": {
                    "type": "string",
                    "description": "The name of the movie to look up streaming availability for.",
                }
            },
            "required": ["movie_title"],
        },
    }
]
function_call = "auto"


llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "temperature": 0,
        "functions": function_defs,
        "function_call": function_call,
    },
)

query_text = """
Tell me more about Avatar movies and find
out what streaming platform it is available on
"""

rag = GraphRAG(retriever=retriever, llm=llm)

result = rag.search(query_text, return_context=True)


def fetch_streaming_info(movie_title: str) -> str:
    platforms = ["Netflix", "Disney+", "AppleTV"]
    return ", ".join(platforms)


if result.function_call:
    func_call = result.function_call
    if func_call["name"] == "get_streaming_availability":
        arguments_str = func_call["arguments"]
        args = json.loads(arguments_str)
        movie_title = args["movie_title"]
        streaming_info = fetch_streaming_info(movie_title)

        second_query = f"""
                {query_text}
                Also, you found that '{movie_title}' is available on {streaming_info}.
                Please provide more details from the knowledge graph but do NOT call any function.
                """

        result_second = rag.search(second_query, return_context=True)
        print("Second-pass answer:", result_second.answer)
else:
    print(result.answer)

driver.close()
