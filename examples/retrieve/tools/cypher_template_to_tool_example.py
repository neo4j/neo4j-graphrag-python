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
Example demonstrating how to use the ToolsRetriever with a Cypher template.

This example shows:
1. How to create a tool from a Cypher template
2. How to use the ToolsRetriever to select and execute tools based on a query
"""

import os
from typing import Any, Callable, Dict, Optional

import neo4j
from dotenv import load_dotenv

from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.retrievers.tools_retriever import ToolsRetriever
from neo4j_graphrag.types import RawSearchResult, RetrieverResultItem

# Load environment variables from .env file (OPENAI_API_KEY required for this example)
load_dotenv()

# Define database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")

# Cypher template to count actors in a specific movie
CYPHER_TEMPLATE = """
MATCH (m:Movie {title: $title})
    OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Actor)
    WITH m, collect(a.name) AS actor_names, count(a) AS actor_count
    RETURN m.title AS movie_title,
        m.plot AS plot,
        m.released AS year,
        actor_count,
        actor_names
"""


class CypherTemplateRetriever(Retriever):
    """
    Custom retriever that executes a parameterized Cypher query template.
    """

    def __init__(
        self,
        driver: neo4j.Driver,
        cypher_template: str,
        neo4j_database: Optional[str] = None,
        result_formatter: Optional[
            Callable[[neo4j.Record], RetrieverResultItem]
        ] = None,
    ):
        """
        Args:
            driver: Neo4j driver instance
            cypher_template: Cypher query with parameters (e.g., "MATCH (m:Movie {title: $title}) RETURN m")
            neo4j_database: Optional database name
            result_formatter: Optional function to format results
        """
        super().__init__(driver, neo4j_database)
        self.cypher_template = cypher_template
        self.result_formatter = result_formatter

    def get_search_results(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        query_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> RawSearchResult:
        """
        Execute the Cypher template with provided parameters.

        Args:
            query_text: Can be used as a parameter (e.g., for movie title)
            query_params: Dictionary of parameters for the Cypher query
            **kwargs: Additional parameters

        Returns:
            RawSearchResult containing neo4j.Record objects
        """
        # Prepare parameters for the Cypher query
        parameters = query_params or {}

        # Optionally use query_text as a parameter
        if query_text and "query_text" not in parameters:
            parameters["query_text"] = query_text

        # Add any additional kwargs as parameters
        parameters.update(kwargs)

        # Execute the query
        try:
            records, summary, keys = self.driver.execute_query(
                self.cypher_template,
                parameters_=parameters,
                database_=self.neo4j_database,
                routing_=neo4j.RoutingControl.READ,
            )

            return RawSearchResult(
                records=records,
                metadata={
                    "cypher_query": self.cypher_template,
                    "parameters": parameters,
                },
            )
        except Exception as e:
            raise RuntimeError(f"Failed to execute Cypher template: {e}") from e


def main() -> None:
    """Run the example."""
    driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
    try:
        neo4j_retriever = CypherTemplateRetriever(
            driver=driver,
            cypher_template=CYPHER_TEMPLATE,
        )

        # Convert retriever to tool
        neo4j_tool = neo4j_retriever.convert_to_tool(
            name="movie_info_tool",
            description=(
                "Retrieves the total number of actors in a specific movie and returns "
                "aggregation metrics including actor count, list of actor names, and movie details. "
                "Use this when the user asks about cast size, number of actors, or movie cast information."
            ),
            parameter_descriptions={
                "query_params": (
                    "Dictionary containing 'title' key with the movie title to analyze. "
                    "Example: {'title': 'The Matrix'}"
                )
            },
        )

        llm = OpenAILLM(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4.1",
            model_params={"temperature": 0.2},
        )

        tools_retriever = ToolsRetriever(
            driver=driver,
            llm=llm,
            tools=[neo4j_tool],
        )

        query_text = "How many actors are there in Around the World in 80 Days?"
        print("Query:", query_text)
        result = tools_retriever.search(query_text=query_text, return_context=True)
        print("Result:", result)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
