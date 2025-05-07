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
Example demonstrating how to convert a retriever to a tool.

This example shows:
1. How to convert a custom StaticRetriever to a Tool
2. How to define parameters for the tool
3. How to execute the tool
"""

import neo4j
from typing import Optional, Any, cast
from unittest.mock import MagicMock

from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult
from neo4j_graphrag.tools.tool import (
    StringParameter,
    ObjectParameter,
)
from neo4j_graphrag.tools.utils import convert_retriever_to_tool


# Create a Retriever that returns static results about Neo4j
# This would illustrate the conversion process of any Retriever (Vector, Hybrid, etc.)
class StaticRetriever(Retriever):
    """A retriever that returns static results about Neo4j."""

    # Disable Neo4j version verification
    VERIFY_NEO4J_VERSION = False

    def __init__(self, driver: neo4j.Driver):
        # Call the parent class constructor with the driver
        super().__init__(driver)

    def get_search_results(
        self, query_text: Optional[str] = None, **kwargs: Any
    ) -> RawSearchResult:
        """Return static information about Neo4j regardless of the query."""
        # Create formatted Neo4j information
        neo4j_info = (
            "# Neo4j Graph Database\n\n"
            "Neo4j is a graph database management system developed by Neo4j, Inc. "
            "It is an ACID-compliant transactional database with native graph storage and processing.\n\n"
            "## Key Features:\n\n"
            "- **Cypher Query Language**: Neo4j's intuitive query language designed specifically for working with graph data\n"
            "- **Property Graphs**: Both nodes and relationships can have properties (key-value pairs)\n"
            "- **ACID Compliance**: Ensures data integrity with full transaction support\n"
            "- **Native Graph Storage**: Optimized storage for graph data structures\n"
            "- **High Availability**: Clustering for enterprise deployments\n"
            "- **Scalability**: Handles billions of nodes and relationships"
        )

        # Create a Neo4j record with the information
        records = [neo4j.Record({"result": neo4j_info})]

        # Return a RawSearchResult with the records and metadata
        return RawSearchResult(records=records, metadata={"query": query_text})


def main() -> None:
    # Convert a StaticRetriever to a tool with specific parameters
    static_retriever = StaticRetriever(driver=cast(Any, MagicMock()))

    # Define parameters for the static retriever tool
    static_parameters = ObjectParameter(
        description="Parameters for the Neo4j information retriever",
        properties={
            "query_text": StringParameter(
                description="The query about Neo4j (any query will return general Neo4j information)",
                required=True,
            ),
        },
    )

    # Convert the retriever to a tool with specific parameters
    static_tool = convert_retriever_to_tool(
        retriever=static_retriever,
        description="Get general information about Neo4j graph database",
        parameters=static_parameters,
        name="Neo4jInfoTool",
    )

    # Print tool information
    print("Example: StaticRetriever with specific parameters")
    print(f"Tool Name: {static_tool.get_name()}")
    print(f"Tool Description: {static_tool.get_description()}")
    print(f"Tool Parameters: {static_tool.get_parameters()}")
    print()

    # Execute the tools (in a real application, this would be done by instructions from an LLM)
    try:
        # Execute the static retriever tool
        print("\nExecuting the static retriever tool...")
        static_result = static_tool.execute(
            query="What is Neo4j?",
        )
        print("Static Search Results:")
        for i, item in enumerate(static_result):
            print(f"{i + 1}. {str(item)[:100]}...")

    except Exception as e:
        print(f"Error executing tool: {e}")


if __name__ == "__main__":
    main()
