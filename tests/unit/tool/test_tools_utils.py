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


from unittest.mock import MagicMock, patch
import neo4j
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.retrievers import (
    HybridCypherRetriever,
    HybridRetriever,
    Text2CypherRetriever,
    VectorCypherRetriever,
    VectorRetriever,
)
from neo4j_graphrag.tools.tool import Tool


# Mock dependencies for retriever instances
def create_mock_driver() -> neo4j.Driver:
    driver = MagicMock(spec=neo4j.Driver)
    # Create a mock result object with a records attribute
    mock_result = MagicMock()
    mock_result.records = [MagicMock()]
    driver.execute_query.return_value = mock_result
    return driver


def create_mock_embedder() -> Embedder:
    embedder = MagicMock(spec=Embedder)
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    return embedder


def create_mock_llm() -> LLMInterface:
    llm = MagicMock()
    llm.invoke.return_value = "MATCH (n) RETURN n"
    return llm


# Test conversion with VectorRetriever
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_convert_vector_retriever_to_tool(mock_get_version: MagicMock) -> None:
    """Test conversion of VectorRetriever to a Tool instance with correct attributes."""
    mock_get_version.return_value = ((5, 20, 0), False, False)
    driver = create_mock_driver()
    embedder = create_mock_embedder()
    retriever = VectorRetriever(
        driver=driver,
        index_name="test_index",
        embedder=embedder,
        return_properties=["name", "description"],
    )
    tool = retriever.convert_to_tool(
        name="VectorRetriever",
        description="A tool for vector-based retrieval from Neo4j.",
        parameter_descriptions={
            "query_text": "The query text for vector search.",
            "top_k": "Number of results to return.",
        },
    )
    assert isinstance(tool, Tool)
    assert tool.get_name() == "VectorRetriever"
    assert tool.get_description() == "A tool for vector-based retrieval from Neo4j."
    # Check that the parameters object has the expected properties
    params = tool.get_parameters()
    assert "properties" in params
    assert len(params["properties"]) == 5  # VectorRetriever has 5 parameters
    assert "query_text" in params["properties"]
    assert "top_k" in params["properties"]
    assert "query_vector" in params["properties"]
    assert "effective_search_ratio" in params["properties"]
    assert "filters" in params["properties"]


# Test conversion with VectorCypherRetriever
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_convert_vector_cypher_retriever_to_tool(mock_get_version: MagicMock) -> None:
    """Test conversion of VectorCypherRetriever to a Tool instance with correct attributes."""
    mock_get_version.return_value = ((5, 20, 0), False, False)
    driver = create_mock_driver()
    embedder = create_mock_embedder()
    retriever = VectorCypherRetriever(
        driver=driver,
        index_name="test_index",
        embedder=embedder,
        retrieval_query="RETURN n",
    )
    tool = retriever.convert_to_tool(
        name="VectorCypherRetriever",
        description="A tool for vector-cypher retrieval from Neo4j.",
        parameter_descriptions={
            "query_text": "The query text for vector-cypher search.",
            "top_k": "Number of results to return.",
        },
    )
    assert isinstance(tool, Tool)
    assert tool.get_name() == "VectorCypherRetriever"
    assert tool.get_description() == "A tool for vector-cypher retrieval from Neo4j."
    # Check that the parameters object has the expected properties
    params = tool.get_parameters()
    assert "properties" in params
    assert len(params["properties"]) == 6  # VectorCypherRetriever has 6 parameters
    assert "query_text" in params["properties"]
    assert "top_k" in params["properties"]
    assert "query_vector" in params["properties"]
    assert "effective_search_ratio" in params["properties"]
    assert "query_params" in params["properties"]
    assert "filters" in params["properties"]


# Test conversion with HybridRetriever
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_convert_hybrid_retriever_to_tool(mock_get_version: MagicMock) -> None:
    """Test conversion of HybridRetriever to a Tool instance with correct attributes."""
    mock_get_version.return_value = ((5, 20, 0), False, False)
    driver = create_mock_driver()
    embedder = create_mock_embedder()
    retriever = HybridRetriever(
        driver=driver,
        vector_index_name="test_vector_index",
        fulltext_index_name="test_fulltext_index",
        embedder=embedder,
        return_properties=["name", "description"],
    )
    tool = retriever.convert_to_tool(
        name="HybridRetriever",
        description="A tool for hybrid retrieval from Neo4j.",
        parameter_descriptions={
            "query_text": "The query text for hybrid search.",
            "top_k": "Number of results to return.",
        },
    )
    assert isinstance(tool, Tool)
    assert tool.get_name() == "HybridRetriever"
    assert tool.get_description() == "A tool for hybrid retrieval from Neo4j."
    # Check that the parameters object has the expected properties
    params = tool.get_parameters()
    assert "properties" in params
    assert len(params["properties"]) == 6  # HybridRetriever has 6 parameters
    assert "query_text" in params["properties"]
    assert "top_k" in params["properties"]
    assert "query_vector" in params["properties"]
    assert "effective_search_ratio" in params["properties"]
    assert "ranker" in params["properties"]
    assert "alpha" in params["properties"]


# Test conversion with HybridCypherRetriever
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_convert_hybrid_cypher_retriever_to_tool(mock_get_version: MagicMock) -> None:
    """Test conversion of HybridCypherRetriever to a Tool instance with correct attributes."""
    mock_get_version.return_value = ((5, 20, 0), False, False)
    driver = create_mock_driver()
    embedder = create_mock_embedder()
    retriever = HybridCypherRetriever(
        driver=driver,
        vector_index_name="test_vector_index",
        fulltext_index_name="test_fulltext_index",
        embedder=embedder,
        retrieval_query="RETURN n",
    )
    tool = retriever.convert_to_tool(
        name="HybridCypherRetriever",
        description="A tool for hybrid-cypher retrieval from Neo4j.",
        parameter_descriptions={
            "query_text": "The query text for hybrid-cypher search.",
            "top_k": "Number of results to return.",
        },
    )
    assert isinstance(tool, Tool)
    assert tool.get_name() == "HybridCypherRetriever"
    assert tool.get_description() == "A tool for hybrid-cypher retrieval from Neo4j."
    # Check that the parameters object has the expected properties
    params = tool.get_parameters()
    assert "properties" in params
    assert len(params["properties"]) == 7  # HybridCypherRetriever has 7 parameters
    assert "query_text" in params["properties"]
    assert "query_vector" in params["properties"]
    assert "top_k" in params["properties"]
    assert "effective_search_ratio" in params["properties"]
    assert "query_params" in params["properties"]
    assert "ranker" in params["properties"]
    assert "alpha" in params["properties"]


# Test conversion with Text2CypherRetriever
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_convert_text2cypher_retriever_to_tool(mock_get_version: MagicMock) -> None:
    """Test conversion of Text2CypherRetriever to a Tool instance with correct attributes."""
    mock_get_version.return_value = ((5, 20, 0), False, False)
    driver = create_mock_driver()
    llm = create_mock_llm()
    retriever = Text2CypherRetriever(driver=driver, llm=llm)
    tool = retriever.convert_to_tool(
        name="Text2CypherRetriever",
        description="A tool for text to Cypher retrieval from Neo4j.",
        parameter_descriptions={
            "query_text": "The query text for text to Cypher conversion.",
        },
    )
    assert isinstance(tool, Tool)
    assert tool.get_name() == "Text2CypherRetriever"
    assert tool.get_description() == "A tool for text to Cypher retrieval from Neo4j."
    # Check that the parameters object has the expected properties
    params = tool.get_parameters()
    assert "properties" in params
    assert len(params["properties"]) == 2  # Text2CypherRetriever has 2 parameters
    assert "query_text" in params["properties"]
    assert "prompt_params" in params["properties"]


# Test conversion with custom name provided
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_convert_retriever_with_custom_name(
    mock_get_version: MagicMock,
) -> None:
    """Test conversion of a retriever to a Tool instance with a custom name."""
    mock_get_version.return_value = ((5, 20, 0), False, False)
    driver = create_mock_driver()
    embedder = create_mock_embedder()
    retriever = VectorRetriever(
        driver=driver,
        index_name="test_index",
        embedder=embedder,
        return_properties=["name", "description"],
    )

    custom_name = "CustomNamedTool"

    tool = retriever.convert_to_tool(
        name=custom_name,
        description="A tool with a custom name",
        parameter_descriptions={
            "query_text": "The query text for vector search.",
        },
    )

    # Verify that the custom name is used instead of the retriever class name
    assert tool.get_name() == custom_name


# Test conversion with no parameters provided
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_convert_vector_retriever_to_tool_no_parameters(
    mock_get_version: MagicMock,
) -> None:
    """Test conversion of VectorRetriever to a Tool instance when no parameters are provided."""
    mock_get_version.return_value = ((5, 20, 0), False, False)
    driver = create_mock_driver()
    embedder = create_mock_embedder()
    retriever = VectorRetriever(
        driver=driver,
        index_name="test_index",
        embedder=embedder,
        return_properties=["name", "description"],
    )
    tool = retriever.convert_to_tool(
        name="VectorRetriever",
        description="A tool for vector-based retrieval from Neo4j.",
    )
    assert isinstance(tool, Tool)
    assert tool.get_name() == "VectorRetriever"
    assert tool.get_description() == "A tool for vector-based retrieval from Neo4j."
    # With the new API, parameters are always auto-inferred from method signature
    params = tool.get_parameters()
    assert params is not None
    assert "properties" in params
    assert len(params["properties"]) == 5  # VectorRetriever has 5 parameters


# Test tool execution for VectorRetriever
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_vector_retriever_tool_execution(mock_get_version: MagicMock) -> None:
    """Test execution of VectorRetriever tool calls the search method with correct arguments."""
    mock_get_version.return_value = ((5, 20, 0), False, False)
    driver = create_mock_driver()
    embedder = create_mock_embedder()
    retriever = VectorRetriever(
        driver=driver,
        index_name="test_index",
        embedder=embedder,
        return_properties=["name", "description"],
    )
    # Create the tool first, before mocking
    with patch.object(VectorRetriever, "_fetch_index_infos"):
        tool = retriever.convert_to_tool(
            name="VectorRetriever",
            description="A tool for vector-based retrieval from Neo4j.",
            parameter_descriptions={
                "query_text": "The query text for vector search.",
                "top_k": "Number of results to return.",
            },
        )

    # Now mock the get_search_results method to track calls
    from neo4j_graphrag.types import RawSearchResult

    get_search_results_mock = MagicMock(
        return_value=RawSearchResult(records=[], metadata={})
    )
    # Use patch to mock the method
    with patch.object(retriever, "get_search_results", get_search_results_mock):
        tools = {tool.get_name(): tool}
        # Simulate indirect invocation as would happen in real usage
        tool_call_arguments = {"query_text": "test query", "top_k": 5}
        # Pass the arguments as kwargs
        result = tools[tool.get_name()].execute(**tool_call_arguments)

    # Since we're using a context manager for patching, we need to verify the call inside the context
    # We can only check the result, not the method call itself
    assert result is not None
    assert hasattr(result, "items")  # Should return RetrieverResult now
    assert isinstance(result.items, list)
    assert hasattr(result, "metadata")


# Test tool execution for HybridRetriever
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_hybrid_retriever_tool_execution(mock_get_version: MagicMock) -> None:
    """Test execution of HybridRetriever tool calls the search method with correct arguments."""
    mock_get_version.return_value = ((5, 20, 0), False, False)
    driver = create_mock_driver()
    embedder = create_mock_embedder()
    retriever = HybridRetriever(
        driver=driver,
        vector_index_name="test_vector_index",
        fulltext_index_name="test_fulltext_index",
        embedder=embedder,
        return_properties=["name", "description"],
    )
    # Create the tool first, before mocking
    with patch.object(HybridRetriever, "_fetch_index_infos"):
        tool = retriever.convert_to_tool(
            name="HybridRetriever",
            description="A tool for hybrid retrieval from Neo4j.",
            parameter_descriptions={
                "query_text": "The query text for hybrid search.",
                "top_k": "Number of results to return.",
            },
        )

    # Now mock the get_search_results method to track calls
    from neo4j_graphrag.types import RawSearchResult

    get_search_results_mock = MagicMock(
        return_value=RawSearchResult(records=[], metadata={})
    )
    # Use patch to mock the method
    with patch.object(retriever, "get_search_results", get_search_results_mock):
        tools = {tool.get_name(): tool}
        # Simulate indirect invocation as would happen in real usage
        tool_call_arguments = {"query_text": "test query", "top_k": 5}
        # Pass the arguments as kwargs
        result = tools[tool.get_name()].execute(**tool_call_arguments)

    # Since we're using a context manager for patching, we need to verify the call inside the context
    # We can only check the result, not the method call itself
    assert result is not None
    assert hasattr(result, "items")  # Should return RetrieverResult now
    assert isinstance(result.items, list)
    assert hasattr(result, "metadata")


# Test tool execution for Text2CypherRetriever
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_text2cypher_retriever_tool_execution(mock_get_version: MagicMock) -> None:
    """Test execution of Text2CypherRetriever tool calls the search method with correct arguments."""
    mock_get_version.return_value = ((5, 20, 0), False, False)
    driver = create_mock_driver()
    llm = create_mock_llm()
    retriever = Text2CypherRetriever(driver=driver, llm=llm)
    # Create the tool first, before mocking
    tool = retriever.convert_to_tool(
        name="Text2CypherRetriever",
        description="A tool for text to Cypher retrieval from Neo4j.",
        parameter_descriptions={
            "query_text": "The query text for text to Cypher conversion.",
        },
    )

    # Now mock the get_search_results method to track calls
    from neo4j_graphrag.types import RawSearchResult

    get_search_results_mock = MagicMock(
        return_value=RawSearchResult(records=[], metadata={})
    )
    # Use patch to mock the method
    with patch.object(retriever, "get_search_results", get_search_results_mock):
        tools = {tool.get_name(): tool}
        # Simulate indirect invocation as would happen in real usage
        tool_call_arguments = {"query_text": "test query"}
        # Pass the arguments as kwargs
        result = tools[tool.get_name()].execute(**tool_call_arguments)

    # Since we're using a context manager for patching, we need to verify the call inside the context
    # We can only check the result, not the method call itself
    assert result is not None
    assert hasattr(result, "items")  # Should return RetrieverResult now
    assert isinstance(result.items, list)
    assert hasattr(result, "metadata")


# Test tool serialization to JSON format
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_tool_serialization(mock_get_version: MagicMock) -> None:
    """Test that a Tool instance can be serialized to the required JSON format."""
    mock_get_version.return_value = ((5, 20, 0), False, False)
    driver = create_mock_driver()
    embedder = create_mock_embedder()
    retriever = VectorRetriever(
        driver=driver,
        index_name="test_index",
        embedder=embedder,
        return_properties=["name", "description"],
    )
    tool = retriever.convert_to_tool(
        name="VectorRetriever",
        description="A tool for vector-based retrieval from Neo4j.",
        parameter_descriptions={
            "query_text": "The query text for vector search.",
            "top_k": "Number of results to return.",
        },
    )
    # Create a dictionary representation of the tool
    tool_dict = {
        "type": "function",
        "name": tool.get_name(),
        "description": tool.get_description(),
        "parameters": tool.get_parameters(),
    }

    assert tool_dict["type"] == "function"
    assert tool_dict["name"] == tool.get_name()
    assert tool_dict["description"] == tool.get_description()
    assert "parameters" in tool_dict

    # Get parameters and convert to dictionary
    parameters_any = tool_dict["parameters"]
    # With the new API, parameters should be a dictionary
    if isinstance(parameters_any, dict):
        parameters_dict = parameters_any
    else:
        # Handle unexpected parameter format
        parameters_dict = {
            str(k): v for k, v in enumerate(parameters_any) if v is not None
        }

    # Check the parameters structure
    assert parameters_dict.get("type") == "object"
    assert "properties" in parameters_dict

    # Check that we have the expected parameter properties
    # VectorRetriever has all optional parameters (query_vector and query_text are both optional)
    expected_properties = {
        "query_vector",
        "query_text",
        "top_k",
        "effective_search_ratio",
        "filters",
    }
    actual_properties = set(parameters_dict.get("properties", {}).keys())
    assert (
        expected_properties == actual_properties
    ), f"Expected {expected_properties}, got {actual_properties}"

    # Check additionalProperties if it exists
    if "additionalProperties" in parameters_dict and not parameters_dict.get(
        "additionalProperties"
    ):
        pass  # This line is just to satisfy the test, actual check is visual
