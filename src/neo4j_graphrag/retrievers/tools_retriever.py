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
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import neo4j

from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult
from neo4j_graphrag.tools.tool import Tool
from neo4j_graphrag.types import LLMMessage


class ToolsRetriever(Retriever):
    """A retriever that uses an LLM to select appropriate tools for retrieval based on user input.

    This retriever takes an LLM instance and a list of Tool objects as input. When a search is performed,
    it uses the LLM to analyze the query and determine which tools (if any) should be used to retrieve
    the necessary data. It then executes the selected tools and returns the combined results.

    Args:
        driver (neo4j.Driver): Neo4j driver instance.
        llm (LLMInterface): LLM instance used to select tools.
        tools (Sequence[Tool]): List of tools available for selection.
        neo4j_database (Optional[str], optional): Neo4j database name. Defaults to None.
        system_instruction (Optional[str], optional): Custom system instruction for the LLM. Defaults to None.
    """

    # Disable Neo4j version verification since this retriever doesn't directly interact with Neo4j
    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        driver: neo4j.Driver,
        llm: LLMInterface,
        tools: Sequence[Tool],
        neo4j_database: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ):
        """Initialize the ToolsRetriever with an LLM and a list of tools."""
        super().__init__(driver, neo4j_database)
        self.llm = llm
        self._tools = list(tools)  # Make a copy to allow modification
        self.system_instruction = (
            system_instruction or self._get_default_system_instruction()
        )

    def _get_default_system_instruction(self) -> str:
        """Get the default system instruction for the LLM."""
        return (
            "You are an assistant that helps select the most appropriate tools to retrieve information "
            "based on the user's query. Analyze the query carefully and determine which tools, if any, "
            "would be most helpful in retrieving the relevant information. You can select multiple tools "
            "if necessary, or none if no tools are appropriate for the query."
        )

    def get_search_results(
        self,
        query_text: str,
        message_history: Optional[List[LLMMessage]] = None,
        **kwargs: Any,
    ) -> RawSearchResult:
        """Use the LLM to select and execute appropriate tools based on the query.

        Args:
            query_text (str): The user's query text.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]], optional):
                Previous conversation history. Defaults to None.
            **kwargs (Any): Additional arguments passed to the tool execution.

        Returns:
            RawSearchResult: The combined results from the executed tools.
        """
        if not self._tools:
            # No tools available, return empty result
            return RawSearchResult(
                records=[],
                metadata={"query": query_text, "error": "No tools available"},
            )

        try:
            # Use the LLM to select appropriate tools
            tool_call_response = self.llm.invoke_with_tools(
                input=query_text,
                tools=self._tools,
                message_history=message_history,
                system_instruction=self.system_instruction,
            )
            # If no tool calls were made, return empty result
            if not tool_call_response.tool_calls:
                return RawSearchResult(
                    records=[],
                    metadata={
                        "query": query_text,
                        "llm_response": tool_call_response.content,
                        "tools_selected": [],
                    },
                )

            # Execute each selected tool and collect results
            all_records = []
            tools_selected = []

            for tool_call in tool_call_response.tool_calls:
                tool_name = tool_call.name
                tools_selected.append(tool_name)

                # Find the tool by name
                selected_tool = next(
                    (tool for tool in self._tools if tool.get_name() == tool_name), None
                )
                if selected_tool is not None:
                    # Extract arguments from the tool call
                    tool_args = tool_call.arguments or {}

                    # Always include the query_text in the arguments for tools that might need it
                    tool_args.setdefault("query", query_text)

                    # Execute the tool with the provided arguments
                    tool_result = selected_tool.execute(**tool_args)
                    # If the tool result is a RawSearchResult, extract its records
                    if hasattr(tool_result, "records"):
                        all_records.extend(tool_result.records)
                    else:
                        # Create a record from the tool result
                        record = neo4j.Record({"result": tool_result})
                        all_records.append(record)

            # Combine metadata from all tool calls
            combined_metadata = {
                "query": query_text,
                "llm_response": tool_call_response.content,
                "tools_selected": tools_selected,
            }

            return RawSearchResult(records=all_records, metadata=combined_metadata)

        except Exception as e:
            # Handle any errors during tool selection or execution
            return RawSearchResult(
                records=[],
                metadata={
                    "query": query_text,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
