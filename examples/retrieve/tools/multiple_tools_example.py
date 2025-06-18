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
#
"""
Example demonstrating how to create multiple domain-specific tools from retrievers.

This example shows:
1. How to create multiple tools from the same retriever type for different use cases
2. How to provide custom parameter descriptions for each tool
3. How type inference works automatically while descriptions are explicit
"""

import neo4j
from typing import cast, Any, Optional
from unittest.mock import MagicMock

from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult


class MockVectorRetriever(Retriever):
    """A mock vector retriever for demonstration purposes."""

    VERIFY_NEO4J_VERSION = False

    def __init__(self, driver: neo4j.Driver, index_name: str):
        super().__init__(driver)
        self.index_name = index_name

    def get_search_results(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        effective_search_ratio: int = 1,
        filters: Optional[dict[str, Any]] = None,
    ) -> RawSearchResult:
        """Get vector search results (mocked for demonstration)."""
        # Return empty results for demo
        return RawSearchResult(records=[], metadata={"index": self.index_name})


def main() -> None:
    """Demonstrate creating multiple domain-specific tools from retrievers."""

    # Create mock driver (in real usage, this would be actual Neo4j driver)
    driver = cast(Any, MagicMock())

    # Create retrievers for different domains using the same retriever type
    # In practice, these would point to different vector indexes

    # Movie recommendations retriever
    movie_retriever = MockVectorRetriever(driver=driver, index_name="movie_embeddings")

    # Product search retriever
    product_retriever = MockVectorRetriever(
        driver=driver, index_name="product_embeddings"
    )

    # Document search retriever
    document_retriever = MockVectorRetriever(
        driver=driver, index_name="document_embeddings"
    )

    # Convert each retriever to a domain-specific tool with custom descriptions

    # 1. Movie recommendation tool
    movie_tool = movie_retriever.convert_to_tool(
        name="movie_search",
        description="Find movie recommendations based on plot, genre, or actor preferences",
        parameter_descriptions={
            "query_text": "Movie title, plot description, genre, or actor name",
            "query_vector": "Pre-computed embedding vector for movie search",
            "top_k": "Number of movie recommendations to return (1-20)",
            "filters": "Optional filters for genre, year, rating, etc.",
            "effective_search_ratio": "Search pool multiplier for better accuracy",
        },
    )

    # 2. Product search tool
    product_tool = product_retriever.convert_to_tool(
        name="product_search",
        description="Search for products matching customer needs and preferences",
        parameter_descriptions={
            "query_text": "Product name, description, or customer need",
            "query_vector": "Pre-computed embedding for product matching",
            "top_k": "Maximum number of product results (1-50)",
            "filters": "Filters for price range, brand, category, availability",
            "effective_search_ratio": "Breadth vs precision trade-off for search",
        },
    )

    # 3. Document search tool
    document_tool = document_retriever.convert_to_tool(
        name="document_search",
        description="Find relevant documents and knowledge articles",
        parameter_descriptions={
            "query_text": "Question, keywords, or topic to search for",
            "query_vector": "Semantic embedding for document retrieval",
            "top_k": "Number of relevant documents to retrieve (1-10)",
            "filters": "Document type, date range, or department filters",
        },
    )

    # Demonstrate that each tool has distinct, meaningful descriptions
    tools = [movie_tool, product_tool, document_tool]

    for tool in tools:
        print(f"\n=== {tool.get_name().upper()} ===")
        print(f"Description: {tool.get_description()}")
        print("Parameters:")

        params = tool.get_parameters()
        for param_name, param_def in params["properties"].items():
            required = (
                "required" if param_name in params.get("required", []) else "optional"
            )
            print(
                f"  - {param_name} ({param_def['type']}, {required}): {param_def['description']}"
            )

    # Show how the same parameter type gets different contextual descriptions
    print("\n=== PARAMETER COMPARISON ===")
    print("Same parameter 'query_text' with different contextual descriptions:")

    for tool in tools:
        params = tool.get_parameters()
        query_text_desc = params["properties"]["query_text"]["description"]
        print(f"  {tool.get_name()}: {query_text_desc}")

    print("\nSame parameter 'top_k' with different contextual descriptions:")
    for tool in tools:
        params = tool.get_parameters()
        top_k_desc = params["properties"]["top_k"]["description"]
        print(f"  {tool.get_name()}: {top_k_desc}")


if __name__ == "__main__":
    main()
