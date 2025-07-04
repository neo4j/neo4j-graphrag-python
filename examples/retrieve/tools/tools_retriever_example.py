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
Example demonstrating how to use the ToolsRetriever.

This example shows:
1. How to create tools from different retrievers
2. How to use the ToolsRetriever to select and execute tools based on a query
"""

import os
from typing import Any, Optional, cast
from unittest.mock import MagicMock
from dotenv import load_dotenv
import requests
from datetime import datetime, date

import neo4j

from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.retrievers.tools_retriever import ToolsRetriever
from neo4j_graphrag.types import RawSearchResult
from neo4j_graphrag.tools.tool import (
    ObjectParameter,
    StringParameter,
    Tool,
)
from neo4j_graphrag.llm.openai_llm import OpenAILLM

# Load environment variables from .env file (OPENAI_API_KEY required for this example)
load_dotenv()


# Create a Retriever that returns static results about Neo4j
class Neo4jInfoRetriever(Retriever):
    """A retriever that returns general information about Neo4j."""

    # Disable Neo4j version verification
    VERIFY_NEO4J_VERSION = False

    def __init__(self, driver: neo4j.Driver):
        # Call the parent class constructor with the driver
        super().__init__(driver)

    def get_search_results(
        self, query_text: Optional[str] = None, **kwargs: Any
    ) -> RawSearchResult:
        """Return general information about Neo4j."""
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


class CalendarTool(Tool):
    """A simple tool to get calendar information."""

    def __init__(self) -> None:
        """Initialize the calendar tool."""
        # Define parameters for the tool
        parameters = ObjectParameter(
            description="Parameters for calendar information retrieval",
            properties={
                "date": StringParameter(
                    description="The date to check events for in YYYY-MM-DD format (e.g., 2025-04-14)",
                ),
            },
            required_properties=["date"],
        )

        # Sample calendar data with fixed dates
        self.calendar_data = {
            "2025-04-15": [
                {"time": "10:00", "title": "Dentist Appointment"},
                {"time": "14:00", "title": "Conference Call"},
            ],
            "2025-04-16": [],
        }

        # Define a wrapper function that handles parameters correctly
        def execute_func(**kwargs: Any) -> str:
            return self.execute_calendar(**kwargs)

        super().__init__(
            name="calendar_tool",
            description="Check calendar events for a specific date in YYYY-MM-DD format",
            parameters=parameters,
            execute_func=execute_func,
        )

    def execute_calendar(self, **kwargs: Any) -> str:
        """Execute the calendar tool.

        Args:
            **kwargs: Dictionary of parameters, including 'date'.

        Returns:
            str: The events for the specified date.
        """
        date = kwargs.get("date")
        if not date:
            return "Error: No date provided"

        # Check for events on the date
        if date in self.calendar_data:
            events_list = self.calendar_data[date]
            if not events_list:
                return f"No events scheduled for {date}"

            events_str = "\n".join(
                f"- {event.get('time', 'All day')}: {event.get('title', 'Untitled event')}"
                for event in events_list
            )
            return f"Events for {date}:\n{events_str}"
        else:
            return f"No events found for {date}"


class WeatherTool(Tool):
    """A tool to fetch weather in Malmö, Sweden for a specific date."""

    def __init__(self) -> None:
        """Initialize the weather tool."""
        parameters = ObjectParameter(
            description="Parameters for fetching weather information about a date.",
            properties={
                "date": StringParameter(
                    description='The date, in YYYY-MM-DD format. Example: "2025-04-25"'
                )
            },
            required_properties=["date"],
        )
        super().__init__(
            name="weather_tool",
            description="Check for weather for a specific date in YYYY-MM-DD format",
            parameters=parameters,
            execute_func=self.execute_weather_retrieval,
        )

    def execute_weather_retrieval(self, **kwargs: Any) -> str:
        """Fetch historical weather data for a given date in Malmö, Sweden."""
        date_str = kwargs.get("date")
        if not date_str:
            return "Error: Date not provided for weather lookup."

        try:
            input_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return f"Error: Invalid date format '{date_str}'. Please use YYYY-MM-DD."

        today_date = date.today()

        if input_date < today_date:
            api_url = "https://archive-api.open-meteo.com/v1/archive"
        else:
            # For today or future dates, use the forecast API
            # Note: Forecast API typically has a limit (e.g., 16 days into the future)
            api_url = "https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": 55.6059,  # Malmö, Sweden
            "longitude": 13.0007,  # Malmö, Sweden
            "start_date": date_str,
            "end_date": date_str,
            "daily": "temperature_2m_max,sunshine_duration",
            "timezone": "Europe/Stockholm",
        }
        headers = {"Accept": "application/json"}

        try:
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            # Try to access keys directly, relying on the existing broader except block for errors
            daily = data["daily"]
            temp_max = daily["temperature_2m_max"][0]
            sunshine_seconds = daily["sunshine_duration"][0]

            sunshine_hours = 0
            if (
                sunshine_seconds is not None
            ):  # API might return null for sunshine_duration
                sunshine_hours = round(sunshine_seconds / 3600, 1)

            return (
                f"Weather for Malmö, Sweden on this day:\n"
                f"- Max Temperature: {temp_max}°C\n"
                f"- Sunshine Duration: {sunshine_hours} hours"
            )
        except requests.exceptions.RequestException as e:
            return f"API request failed for weather data: {e}"
        except (
            ValueError,
            KeyError,
        ) as e:
            return f"Error parsing weather data for Malmö on {date_str}: {e}"


def main() -> None:
    """Run the example."""
    # Create a mock Neo4j driver
    driver = cast(neo4j.Driver, MagicMock())

    # Create retrievers
    neo4j_retriever = Neo4jInfoRetriever(driver=driver)

    # Convert retrievers to tools
    neo4j_tool = neo4j_retriever.convert_to_tool(
        name="neo4j_info_tool",
        description="Get information about Neo4j graph database",
        parameter_descriptions={
            "query_text": "The query about Neo4j",
        },
    )

    # Create a calendar tool
    calendar_tool = CalendarTool()

    # Create a weather tool
    weather_tool = WeatherTool()

    # Create an OpenAI LLM
    llm = OpenAILLM(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o",
        model_params={
            "temperature": 0.2,
        },
    )

    # Print tool information for debugging
    print("\nTool Information:")
    print(f"Neo4j Tool: {neo4j_tool.get_name()}, {neo4j_tool.get_description()}")
    print(
        f"Calendar Tool: {calendar_tool.get_name()}, {calendar_tool.get_description()}"
    )
    parameters_description = (
        weather_tool._parameters.description
        if weather_tool._parameters
        else "No parameters description"
    )
    print(
        f"Weather Tool: {weather_tool.get_name()}, {weather_tool.get_description()}: {parameters_description}"
    )

    # Create a ToolsRetriever with the LLM and tools
    tools_retriever = ToolsRetriever(
        driver=driver,
        llm=llm,
        tools=[neo4j_tool, calendar_tool, weather_tool],
    )

    # Test queries
    test_queries = [
        "What is Neo4j?",
        "Do I have any meetings the 15th of April 2025?",
        "Any information about 2025-04-16?",
    ]

    # Run just the tools retriever directly to show metadata etc.
    print(f"\n\n{'=' * 80}")
    print("Retriever call examples, to show metadata etc.")
    print(f"{'=' * 80}")
    for query in test_queries:
        print(f"Query: {query}")

        try:
            # Get search results through the ToolsRetriever
            result = tools_retriever.get_search_results(query_text=query)

            # Print metadata
            if result.metadata is not None:
                print(f"\nTools selected: {result.metadata.get('tools_selected', [])}")
                if result.metadata.get("error", ""):
                    print(f"Error: {result.metadata.get('error', '')}")

            # Print results
            print("\nRESULTS:")
            for i, record in enumerate(result.records):
                print(f"\n--- Result {i + 1} ---")
                print(f"Content: {record.get('content', 'N/A')}")
                print(f"Tool: {record.get('tool_name', 'Unknown')}")
                if record.get("metadata"):
                    print(f"Metadata: {record.get('metadata')}")
        except Exception as e:
            print(f"Error: {str(e)}")
        print(f"{'=' * 80}")

    # For demo purposes, run the queries through GraphRAG to get text input -> text output
    print(f"\n\n{'=' * 80}")
    print("Full GraphRAG examples")
    print(f"{'=' * 80}")
    for query in test_queries:
        print(f"Query: {query}")
        # Full GraphRAG example
        graphrag = GraphRAG(
            llm=llm,
            retriever=tools_retriever,
        )
        rag_result = graphrag.search(query_text=query, return_context=False)
        print(f"Answer: {rag_result.answer}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
