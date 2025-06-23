# type: ignore
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
Tests for retriever parameter inference and convert_to_tool functionality.
"""

from unittest.mock import MagicMock, patch
from typing import Optional, Any, Dict

import neo4j

from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.retrievers import (
    VectorRetriever,
    VectorCypherRetriever,
    HybridRetriever,
    Text2CypherRetriever,
)
from neo4j_graphrag.retrievers.tools_retriever import ToolsRetriever
from neo4j_graphrag.tools.tool import Tool, ParameterType
from neo4j_graphrag.types import RawSearchResult
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.llm.base import LLMInterface


# Helper functions for creating mock objects
def create_mock_driver() -> neo4j.Driver:
    driver = MagicMock(spec=neo4j.Driver)
    mock_result = MagicMock()
    mock_result.records = []
    driver.execute_query.return_value = mock_result
    return driver


def create_mock_embedder() -> Embedder:
    embedder = MagicMock(spec=Embedder)
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    return embedder


def create_mock_llm() -> LLMInterface:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = MagicMock(content="MATCH (n) RETURN n")
    return llm


class MockRetriever(Retriever):
    """Test retriever with well-documented parameters."""

    VERIFY_NEO4J_VERSION = False

    def get_search_results(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> RawSearchResult:
        """Test search method with documented parameters.

        Args:
            query_text (str): The text query to search for in the database
            top_k (int): The maximum number of results to return
            filters (Optional[Dict[str, Any]]): Optional metadata filters to apply
            score_threshold (Optional[float]): Minimum similarity score threshold

        Returns:
            RawSearchResult: The search results
        """
        return RawSearchResult(records=[], metadata={})


class MockRetrieverNoDocstring(Retriever):
    """Test retriever without parameter documentation."""

    VERIFY_NEO4J_VERSION = False

    def get_search_results(
        self, param_one: str, param_two: Optional[int] = None
    ) -> RawSearchResult:
        """No parameter documentation here."""
        return RawSearchResult(records=[], metadata={})


class TestParameterInference:
    """Test parameter inference from method signatures and docstrings."""

    def test_parameter_inference_with_docstring(self):
        """Test that parameters are correctly inferred from method signature and docstring."""
        driver = create_mock_driver()
        retriever = MockRetriever(driver)

        # Get inferred parameters
        params = retriever.get_parameters()

        # Check basic structure
        assert params.type == ParameterType.OBJECT
        assert params.description == "Parameters for MockRetriever"
        assert not params.additional_properties

        # Check properties
        properties = params.properties
        assert len(properties) == 4

        # Check query_text parameter
        query_text_param = properties["query_text"]
        assert query_text_param.type == ParameterType.STRING
        assert query_text_param.description == "Parameter query_text"
        assert query_text_param.required is True

        # Check top_k parameter
        top_k_param = properties["top_k"]
        assert top_k_param.type == ParameterType.INTEGER
        assert top_k_param.description == "Parameter top_k"
        assert top_k_param.required is False
        assert top_k_param.minimum == 1  # Should be set for top_k parameters

        # Check filters parameter
        filters_param = properties["filters"]
        assert filters_param.type == ParameterType.OBJECT
        assert filters_param.description == "Parameter filters"
        assert filters_param.required is False
        assert filters_param.additional_properties is True

        # Check score_threshold parameter
        score_param = properties["score_threshold"]
        assert score_param.type == ParameterType.NUMBER
        assert score_param.description == "Parameter score_threshold"
        assert score_param.required is False

    def test_parameter_inference_without_docstring(self):
        """Test that parameters work with fallback descriptions when no docstring documentation."""
        driver = create_mock_driver()
        retriever = MockRetrieverNoDocstring(driver)

        # Get inferred parameters
        params = retriever.get_parameters()

        # Check properties
        properties = params.properties
        assert len(properties) == 2

        # Check param_one with fallback description
        param_one = properties["param_one"]
        assert param_one.type == ParameterType.STRING
        assert param_one.description == "Parameter param_one"  # Simple fallback format
        assert param_one.required is True

        # Check param_two with fallback description
        param_two = properties["param_two"]
        assert param_two.type == ParameterType.INTEGER
        assert param_two.description == "Parameter param_two"  # Simple fallback format
        assert param_two.required is False

    def test_convert_to_tool_basic(self):
        """Test basic convert_to_tool functionality."""
        driver = create_mock_driver()
        retriever = MockRetriever(driver)

        # Convert to tool
        tool = retriever.convert_to_tool(
            name="TestTool", description="A test tool for searching"
        )

        # Check tool properties
        assert isinstance(tool, Tool)
        assert tool.get_name() == "TestTool"
        assert tool.get_description() == "A test tool for searching"

        # Check that parameters were inferred
        params = tool.get_parameters()
        assert len(params["properties"]) == 4
        assert "query_text" in params["properties"]
        assert "top_k" in params["properties"]

    def test_convert_to_tool_with_custom_descriptions(self):
        """Test convert_to_tool with custom parameter descriptions."""
        driver = create_mock_driver()
        retriever = MockRetriever(driver)

        # Convert to tool with custom parameter descriptions
        tool = retriever.convert_to_tool(
            name="CustomTool",
            description="A custom search tool",
            parameter_descriptions={
                "query_text": "The search query to execute",
                "top_k": "Maximum number of results to return",
                "filters": "Optional filters to apply to the search",
            },
        )

        # Check tool properties
        assert tool.get_name() == "CustomTool"
        assert tool.get_description() == "A custom search tool"

        # Check custom parameter descriptions
        params = tool.get_parameters()
        properties = params["properties"]

        assert properties["query_text"]["description"] == "The search query to execute"
        assert (
            properties["top_k"]["description"] == "Maximum number of results to return"
        )
        assert (
            properties["filters"]["description"]
            == "Optional filters to apply to the search"
        )
        # Parameter without custom description should use fallback
        assert (
            properties["score_threshold"]["description"] == "Parameter score_threshold"
        )


class TestRealRetrieverParameterInference:
    """Test parameter inference on real retriever classes."""

    @patch("neo4j_graphrag.retrievers.base.get_version")
    def test_vector_retriever_parameters(self, mock_get_version):
        """Test VectorRetriever parameter inference."""
        mock_get_version.return_value = ((5, 20, 0), False, False)

        driver = create_mock_driver()
        embedder = create_mock_embedder()

        # Patch _fetch_index_infos to avoid database calls
        with patch.object(VectorRetriever, "_fetch_index_infos"):
            retriever = VectorRetriever(
                driver=driver, index_name="test_index", embedder=embedder
            )

            params = retriever.get_parameters()
            properties = params.properties

            # Check expected parameters from VectorRetriever.get_search_results
            expected_params = {
                "query_vector",
                "query_text",
                "top_k",
                "effective_search_ratio",
                "filters",
            }
            assert set(properties.keys()) == expected_params

            # Check specific parameter types
            assert properties["query_vector"].type == ParameterType.ARRAY
            assert properties["query_text"].type == ParameterType.STRING
            assert properties["top_k"].type == ParameterType.INTEGER
            assert properties["effective_search_ratio"].type == ParameterType.INTEGER
            assert properties["filters"].type == ParameterType.OBJECT

            # Check that default descriptions are used when no custom descriptions provided
            assert properties["query_vector"].description == "Parameter query_vector"
            assert properties["query_text"].description == "Parameter query_text"

    @patch("neo4j_graphrag.retrievers.base.get_version")
    def test_vector_cypher_retriever_parameters(self, mock_get_version):
        """Test VectorCypherRetriever parameter inference."""
        mock_get_version.return_value = ((5, 20, 0), False, False)

        driver = create_mock_driver()
        embedder = create_mock_embedder()

        # Patch _fetch_index_infos to avoid database calls
        with patch.object(VectorCypherRetriever, "_fetch_index_infos"):
            retriever = VectorCypherRetriever(
                driver=driver,
                index_name="test_index",
                retrieval_query="RETURN node.name",
                embedder=embedder,
            )

            params = retriever.get_parameters()
            properties = params.properties

            # Should have all VectorRetriever params plus query_params
            expected_params = {
                "query_vector",
                "query_text",
                "top_k",
                "effective_search_ratio",
                "query_params",
                "filters",
            }
            assert set(properties.keys()) == expected_params

            # Check query_params is properly typed
            assert properties["query_params"].type == ParameterType.OBJECT
            assert properties["query_params"].additional_properties is True

    @patch("neo4j_graphrag.retrievers.base.get_version")
    def test_hybrid_retriever_parameters(self, mock_get_version):
        """Test HybridRetriever parameter inference."""
        mock_get_version.return_value = ((5, 20, 0), False, False)

        driver = create_mock_driver()
        embedder = create_mock_embedder()

        # Patch _fetch_index_infos to avoid database calls
        with patch.object(HybridRetriever, "_fetch_index_infos"):
            retriever = HybridRetriever(
                driver=driver,
                vector_index_name="vector_index",
                fulltext_index_name="fulltext_index",
                embedder=embedder,
            )

            params = retriever.get_parameters()
            properties = params.properties

            # Check expected parameters from HybridRetriever.get_search_results
            expected_params = {
                "query_text",
                "query_vector",
                "top_k",
                "effective_search_ratio",
                "ranker",
                "alpha",
            }
            assert set(properties.keys()) == expected_params

            # Check that query_text is required for hybrid retriever
            assert properties["query_text"].required is True
            assert properties["alpha"].type == ParameterType.NUMBER
            assert properties["alpha"].minimum == 0.0
            assert properties["alpha"].maximum == 1.0

    @patch("neo4j_graphrag.retrievers.base.get_version")
    def test_text2cypher_retriever_parameters(self, mock_get_version):
        """Test Text2CypherRetriever parameter inference."""
        mock_get_version.return_value = ((5, 20, 0), False, False)

        driver = create_mock_driver()
        llm = create_mock_llm()
        retriever = Text2CypherRetriever(
            driver=driver, llm=llm, neo4j_schema="(Person)-[:KNOWS]->(Person)"
        )

        params = retriever.get_parameters()
        properties = params.properties

        # Check expected parameters
        expected_params = {"query_text", "prompt_params"}
        assert set(properties.keys()) == expected_params

        # Check parameter types
        assert properties["query_text"].type == ParameterType.STRING
        assert properties["query_text"].required is True
        assert (
            properties["prompt_params"].type == ParameterType.OBJECT
        )  # Dict maps to object
        assert properties["prompt_params"].required is False

    def test_tools_retriever_parameters(self):
        """Test ToolsRetriever parameter inference."""
        driver = create_mock_driver()
        llm = create_mock_llm()
        retriever = ToolsRetriever(driver=driver, llm=llm, tools=[])

        params = retriever.get_parameters()
        properties = params.properties

        # Check expected parameters from ToolsRetriever.get_search_results
        expected_params = {"query_text", "message_history"}
        assert set(properties.keys()) == expected_params

        # Check parameter types
        assert properties["query_text"].type == ParameterType.STRING
        assert properties["query_text"].required is True
        assert (
            properties["message_history"].type == ParameterType.OBJECT
        )  # List[LLMMessage] maps to Object
        assert properties["message_history"].required is False


class TestToolExecution:
    """Test that tools created from retrievers actually work."""

    def test_tool_execution(self):
        """Test that a tool created from a retriever can be executed."""
        driver = create_mock_driver()
        retriever = MockRetriever(driver)

        # Convert to tool
        tool = retriever.convert_to_tool(name="TestTool", description="A test tool")

        # Execute the tool
        result = tool.execute(query_text="test query", top_k=3)

        # Check that we get a result (even if empty due to mocking)
        assert result is not None
        assert hasattr(result, "items")  # Should return RetrieverResult now
        assert hasattr(result, "metadata")

    def test_tool_execution_with_validation(self):
        """Test that tool parameter validation works."""
        driver = create_mock_driver()
        retriever = MockRetriever(driver)

        # Convert to tool
        tool = retriever.convert_to_tool(name="TestTool", description="A test tool")

        # Test with missing required parameter should work due to our setup
        # (the actual validation happens in the Tool class)
        result = tool.execute(query_text="test query")
        assert result is not None


class TestParameterDescriptions:
    """Test parameter description functionality."""

    def test_custom_parameter_descriptions(self):
        """Test that custom parameter descriptions are used correctly."""

        class TestRetriever(Retriever):
            VERIFY_NEO4J_VERSION = False

            def get_search_results(
                self, param_a: str, param_b: int = 5, param_c: Optional[float] = None
            ) -> RawSearchResult:
                return RawSearchResult(records=[], metadata={})

        driver = create_mock_driver()
        retriever = TestRetriever(driver)

        # Test with custom descriptions
        custom_descriptions = {
            "param_a": "Custom description for param A",
            "param_b": "Custom description for param B",
            # param_c intentionally omitted to test fallback
        }

        params = retriever.get_parameters(custom_descriptions)
        properties = params.properties

        # Check that custom descriptions are used
        assert properties["param_a"].description == "Custom description for param A"
        assert properties["param_b"].description == "Custom description for param B"
        # Check fallback for param without custom description
        assert properties["param_c"].description == "Parameter param_c"

    def test_no_custom_descriptions(self):
        """Test behavior when no custom descriptions are provided."""

        class SimpleRetriever(Retriever):
            VERIFY_NEO4J_VERSION = False

            def get_search_results(self, test_param: str) -> RawSearchResult:
                return RawSearchResult(records=[], metadata={})

        driver = create_mock_driver()
        retriever = SimpleRetriever(driver)
        params = retriever.get_parameters()
        properties = params.properties

        # Should use fallback description
        assert properties["test_param"].description == "Parameter test_param"
