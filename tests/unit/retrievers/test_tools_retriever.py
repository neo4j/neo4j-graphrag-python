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

# Standard library imports
from typing import Any, List, cast
from unittest.mock import MagicMock

import neo4j

# Local imports
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import ToolCall, ToolCallResponse
from neo4j_graphrag.retrievers.tools_retriever import ToolsRetriever
from neo4j_graphrag.tools.tool import Tool


# Mock dependencies
def create_mock_driver() -> neo4j.Driver:
    driver = MagicMock(spec=neo4j.Driver)
    # Create a mock result object with a records attribute
    mock_result = MagicMock()
    mock_result.records = [MagicMock()]
    driver.execute_query.return_value = mock_result
    return cast(neo4j.Driver, driver)


def create_mock_llm() -> Any:
    llm = MagicMock(spec=LLMInterface)
    return llm


def create_mock_tool(name: str = "MockTool", description: str = "A mock tool") -> Any:
    tool = MagicMock(spec=Tool)
    cast(Any, tool.get_name).return_value = name
    cast(Any, tool.get_description).return_value = description
    cast(Any, tool.get_parameters).return_value = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to search for",
            }
        },
    }
    # Mock the execute method to return a dictionary with records and metadata
    cast(Any, tool.execute).return_value = {
        "records": [neo4j.Record({"result": f"Result from {name}"})],
        "metadata": {"source": name},
    }
    return tool


class TestToolsRetriever:
    """Test the ToolsRetriever class."""

    def test_initialization(self) -> None:
        """Test that the ToolsRetriever initializes correctly."""
        driver = create_mock_driver()
        llm = create_mock_llm()
        tools = [create_mock_tool("Tool1"), create_mock_tool("Tool2")]

        retriever = ToolsRetriever(driver=driver, llm=llm, tools=tools)

        assert retriever.llm == llm
        assert len(retriever._tools) == 2
        assert retriever._tools[0].get_name() == "Tool1"
        assert retriever._tools[1].get_name() == "Tool2"

    def test_get_search_results_no_tools(self) -> None:
        """Test that get_search_results returns an empty result when no tools are available."""
        driver = create_mock_driver()
        llm = create_mock_llm()
        tools: List[Tool] = []

        retriever = ToolsRetriever(driver=driver, llm=llm, tools=tools)
        result = retriever.get_search_results(query_text="Test query")

        assert result.records == []
        assert result.metadata is not None
        assert result.metadata.get("query") == "Test query"
        assert "error" in result.metadata
        assert result.metadata.get("error") == "No tools available"

    def test_get_search_results_no_tool_calls(self) -> None:
        """Test that get_search_results returns an empty result when the LLM doesn't select any tools."""
        driver = create_mock_driver()
        llm = create_mock_llm()
        tools = [create_mock_tool("Tool1"), create_mock_tool("Tool2")]

        # Mock the LLM to return a response with no tool calls
        cast(Any, llm.invoke_with_tools).return_value = ToolCallResponse(
            content="I don't need any tools for this query.",
            tool_calls=[],
        )

        retriever = ToolsRetriever(driver=driver, llm=llm, tools=tools)
        result = retriever.get_search_results(query_text="Test query")

        assert result.records == []
        assert result.metadata is not None
        assert result.metadata.get("query") == "Test query"
        assert (
            result.metadata.get("llm_response")
            == "I don't need any tools for this query."
        )
        assert result.metadata.get("tools_selected") == []

    def test_get_search_results_with_tool_calls(self) -> None:
        """Test that get_search_results correctly executes selected tools and returns their results."""
        driver = create_mock_driver()
        llm = create_mock_llm()
        tool1 = create_mock_tool("Tool1")
        tool2 = create_mock_tool("Tool2")
        tools = [tool1, tool2]

        # Mock the LLM to return a response with tool calls
        cast(Any, llm.invoke_with_tools).return_value = ToolCallResponse(
            content="I'll use Tool1 for this query.",
            tool_calls=[
                ToolCall(
                    name="Tool1",
                    arguments={"query": "Test query"},
                )
            ],
        )

        # Mock the tool execution to return a simple string value
        # This is processed by the ToolsRetriever and converted to a neo4j.Record
        cast(Any, tool1).execute.return_value = "Result from Tool1"

        retriever = ToolsRetriever(driver=driver, llm=llm, tools=tools)
        result = retriever.get_search_results(query_text="Test query")

        # Check that the LLM was called with the right arguments
        cast(Any, llm.invoke_with_tools).assert_called_once_with(
            input="Test query",
            tools=tools,
            message_history=None,
            system_instruction=retriever.system_instruction,
        )

        # Check that the tool was executed with the right arguments
        tool1.execute.assert_called_once_with(query="Test query")

        # Check that the result contains the expected records and metadata
        assert len(result.records) == 1
        # The record is a neo4j.Record object
        assert isinstance(result.records[0], neo4j.Record)
        # Access the result directly using index 0
        assert result.records[0][0] == "Result from Tool1"
        assert result.metadata is not None
        assert result.metadata.get("query") == "Test query"
        assert result.metadata.get("llm_response") == "I'll use Tool1 for this query."
        assert result.metadata.get("tools_selected") == ["Tool1"]

    def test_get_search_results_with_multiple_tool_calls(self) -> None:
        """Test that get_search_results correctly executes multiple selected tools and combines their results."""
        driver = create_mock_driver()
        llm = create_mock_llm()
        tool1 = create_mock_tool("Tool1")
        tool2 = create_mock_tool("Tool2")
        tools = [tool1, tool2]

        # Mock the LLM to return a response with multiple tool calls
        cast(Any, llm.invoke_with_tools).return_value = ToolCallResponse(
            content="I'll use both Tool1 and Tool2 for this query.",
            tool_calls=[
                ToolCall(
                    name="Tool1",
                    arguments={"query": "Test query part 1"},
                ),
                ToolCall(
                    name="Tool2",
                    arguments={"query": "Test query part 2"},
                ),
            ],
        )

        # Mock the tool executions to return specific records
        tool1_record = neo4j.Record({"result": "Result from Tool1"})
        cast(Any, tool1.execute).return_value = {
            "records": [tool1_record],
            "metadata": {"source": "Tool1"},
        }

        tool2_record = neo4j.Record({"result": "Result from Tool2"})
        cast(Any, tool2.execute).return_value = {
            "records": [tool2_record],
            "metadata": {"source": "Tool2"},
        }

        retriever = ToolsRetriever(driver=driver, llm=llm, tools=tools)
        result = retriever.get_search_results(query_text="Test query")

        # Check that both tools were executed with the right arguments
        cast(Any, tool1.execute).assert_called_once_with(query="Test query part 1")
        cast(Any, tool2.execute).assert_called_once_with(query="Test query part 2")

        # Check that the result contains the expected records and metadata
        assert len(result.records) == 2
        assert result.metadata is not None
        assert result.metadata.get("query") == "Test query"
        assert (
            result.metadata.get("llm_response")
            == "I'll use both Tool1 and Tool2 for this query."
        )
        assert result.metadata.get("tools_selected") == ["Tool1", "Tool2"]

    def test_get_search_results_with_error(self) -> None:
        """Test that get_search_results handles errors during tool execution."""
        driver = create_mock_driver()
        llm = create_mock_llm()
        tool = create_mock_tool("Tool1")
        tools = [tool]

        # Mock the LLM to raise an exception
        cast(Any, llm.invoke_with_tools).side_effect = Exception("LLM error")

        retriever = ToolsRetriever(driver=driver, llm=llm, tools=tools)
        result = retriever.get_search_results(query_text="Test query")

        # Check that the result contains the error information
        assert result.records == []
        assert result.metadata is not None
        assert result.metadata.get("query") == "Test query"
        assert result.metadata.get("error") == "LLM error"
        assert result.metadata.get("error_type") == "Exception"

    def test_custom_system_instruction(self) -> None:
        """Test that a custom system instruction is used when provided."""
        driver = create_mock_driver()
        llm = create_mock_llm()
        tools = [create_mock_tool("Tool1")]
        custom_instruction = "This is a custom system instruction."

        retriever = ToolsRetriever(
            driver=driver, llm=llm, tools=tools, system_instruction=custom_instruction
        )

        assert retriever.system_instruction == custom_instruction

        # Test that the custom instruction is passed to the LLM
        retriever.get_search_results(query_text="Test query")

        llm.invoke_with_tools.assert_called_once_with(
            input="Test query",
            tools=tools,
            message_history=None,
            system_instruction=custom_instruction,
        )
