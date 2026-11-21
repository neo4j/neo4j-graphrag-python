#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
The example sends each question to the retriever that suits it, with the ToolsRetriever.

- The DecisionWalkRetriever answers questions that follow relationships from a named
  entity, e.g., "Who directed the movies Hugo Weaving acted in?". A decision model
  chooses each step, so no LLM is used to retrieve.
- The Text2CypherRetriever answers the questions that a walk can't, e.g., questions that
  count, average, rank or filter by property values, or that have no named starting point.

An LLM reads the question and the two tool descriptions, and it picks the tool.

Requires OPENAI_API_KEY and OPENROUTER_API_KEY in the environment (or a .env file), and
`pip install "neo4j-graphrag[openai,openrouter]"`.
"""

import neo4j
from dotenv import load_dotenv

from neo4j_graphrag.decision import OpenRouterDecisionModel
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import (
    DecisionWalkRetriever,
    Text2CypherRetriever,
    ToolsRetriever,
)
from neo4j_graphrag.tool import ObjectParameter, StringParameter, Tool

load_dotenv()

# Neo4j's public, read-only demo database of movies, people and ratings
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"

QUESTION_ONLY = ObjectParameter(
    description="Parameters for the search",
    properties={"query_text": StringParameter(description="The user's question")},
    required_properties=["query_text"],
    additional_properties=False,
)

llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
cypher_llm = OpenAILLM(model_name="gpt-4.1", model_params={"temperature": 0})
decision_model = OpenRouterDecisionModel(
    "liquid/d1-ultra", provider={"order": ["liquid"], "allow_fallbacks": False}
)

with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
    # 1. The two retrievers
    walk = DecisionWalkRetriever(
        driver,
        decision_model,
        fulltext_index_name="personFulltext",  # finds the person that the question is about
        node_name_properties=["name", "title"],
        neo4j_database=DATABASE,
    )
    text2cypher = Text2CypherRetriever(driver, cypher_llm, neo4j_database=DATABASE)

    # 2. Each becomes a tool. The description is what the LLM uses to choose, so it says
    #    plainly which questions each tool is for.
    walk_tool = Tool(
        name="follow_relationships",
        description=(
            "Finds things connected to a specific, named person by following relationships "
            "in the movie graph, one or more steps. Use it for questions that start from a "
            "named person, such as 'Which movies did Hugo Weaving act in?' or 'Who directed "
            "the movies Tom Hanks acted in?'. Do not use it for counting, averages, rankings, "
            "sorting, or filtering by numbers or dates."
        ),
        parameters=QUESTION_ONLY,
        execute_func=lambda query_text: walk.search(query_text=query_text),
    )
    cypher_tool = Tool(
        name="query_database",
        description=(
            "Writes and runs a database query. Use it for questions that count, average, "
            "rank, sort or filter by property values, or that don't start from one named "
            "entity, such as 'How many movies did Tom Hanks act in?' or 'What are the 5 "
            "highest-rated movies?'."
        ),
        parameters=QUESTION_ONLY,
        execute_func=lambda query_text: text2cypher.search(query_text=query_text),
    )

    # 3. The ToolsRetriever lets the LLM pick a tool for each question, and GraphRAG
    #    writes the answer
    router = ToolsRetriever(driver=driver, llm=llm, tools=[walk_tool, cypher_tool])
    rag = GraphRAG(retriever=router, llm=llm)

    for question in [
        "Which movies did Hugo Weaving act in?",
        "Who directed the movies that Hugo Weaving acted in?",
        "How many movies did Tom Hanks act in?",
        "What are the 5 highest-rated movies by IMDb rating?",
    ]:
        result = rag.search(query_text=question, return_context=True)
        context = result.retriever_result
        tools = (context.metadata or {}).get("tools_selected") if context else None
        print(f"\nQ: {question}\n   tool: {tools}\n   A: {result.answer}")

decision_model.close()
