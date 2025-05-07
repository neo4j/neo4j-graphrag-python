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

from typing import Any, Dict, Optional, Union

from neo4j_graphrag.tools.tool import Tool, ObjectParameter


def convert_retriever_to_tool(
    retriever: Any,
    description: Optional[str] = None,
    parameters: Optional[Union[ObjectParameter, Dict[str, Any]]] = None,
    name: Optional[str] = None,
) -> Tool:
    """Convert a retriever instance to a Tool object.

    Args:
        retriever (Any): The retriever instance to convert.
        description (Optional[str]): Custom description for the tool. If not provided,
            an attempt will be made to infer it from the retriever or a generic description will be used.
        parameters (Optional[Union[ObjectParameter, Dict[str, ToolParameter]]]): Custom parameters for the tool.
            If not provided, no parameters will be included in the tool.
        name (Optional[str]): Custom name for the tool. If not provided,
            an attempt will be made to infer it from the retriever or a default name will be used.

    Returns:
        RetrieverTool: A Tool object configured to use the retriever's search functionality.
    """
    # Use provided name or infer it from the retriever
    if name is None:
        name = getattr(retriever, "name", None) or getattr(
            retriever.__class__, "__name__", "UnnamedRetrieverTool"
        )

    # Infer description if not provided
    if description is None:
        description = (
            getattr(retriever, "description", None)
            or f"A tool for retrieving data using {name}."
        )

    # Parameters can be None

    # Define a function that matches the Callable[[str, ...], Any] signature
    def execute_func(**kwargs: Any) -> Any:
        # The retriever's get_search_results method is expected to handle
        # arguments like query_text, top_k, etc., passed as keyword arguments.
        # The Tool's 'parameters' definition (e.g., ObjectParameter) ensures
        # that these arguments are provided in kwargs when Tool.execute is called.
        return retriever.get_search_results(**kwargs)

    # Ensure name is a string
    tool_name = str(name) if name is not None else "UnnamedRetrieverTool"

    # Create a Tool object from the retriever

    # Pass parameters directly to the Tool constructor
    # If parameters is None, the Tool class will handle it appropriately
    return Tool(
        name=tool_name,
        description=description,
        execute_func=execute_func,
        parameters=parameters,
    )
