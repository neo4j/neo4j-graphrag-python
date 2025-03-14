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

import unittest
from unittest.mock import MagicMock, patch

import pytest
import neo4j
from neo4j import Record

from neo4j_graphrag.exceptions import (
    RetrieverInitializationError,
    SearchValidationError,
)
from neo4j_graphrag.retrievers.cypher import CypherRetriever
from neo4j_graphrag.types import RetrieverResultItem


class TestCypherRetriever(unittest.TestCase):
    # Define class attributes for mypy
    patcher1: unittest.mock._patch[MagicMock]
    patcher2: unittest.mock._patch[bool]
    mock_check_driver: MagicMock
    
    @classmethod
    def setUpClass(cls) -> None:
        # Patch the Neo4jDriverModel.check_driver method to pass validation with MagicMock
        cls.patcher1 = patch("neo4j_graphrag.types.Neo4jDriverModel.check_driver")
        cls.mock_check_driver = cls.patcher1.start()
        cls.mock_check_driver.side_effect = lambda v: v

        # Patch the version check in the Retriever base class to avoid Neo4j version validation
        cls.patcher2 = patch(
            "neo4j_graphrag.retrievers.base.Retriever.VERIFY_NEO4J_VERSION", False
        )
        cls.patcher2.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.patcher1.stop()
        cls.patcher2.stop()

    def setUp(self) -> None:
        # Create a mock driver
        self.driver = MagicMock(spec=neo4j.Driver)
        self.driver.execute_query.return_value = (
            [Record({"m": {"title": "Test Movie"}, "score": 0.9})],
            None,
            None,
        )

        # Sample query and parameters
        self.valid_query = "MATCH (m:Movie {title: $movie_title}) RETURN m"
        self.valid_parameters = {
            "movie_title": {"type": "string", "description": "Title of a movie"}
        }

    def test_init_success(self) -> None:
        # Test successful initialization
        retriever = CypherRetriever(
            driver=self.driver,
            query=self.valid_query,
            parameters=self.valid_parameters,
        )
        assert retriever.query == self.valid_query
        assert "movie_title" in retriever.parameters

    def test_init_empty_query(self) -> None:
        # Test initialization with empty query
        with pytest.raises(RetrieverInitializationError):
            CypherRetriever(
                driver=self.driver,
                query="",
                parameters=self.valid_parameters,
            )

    def test_init_invalid_query(self) -> None:
        # Test initialization with invalid query
        with pytest.raises(RetrieverInitializationError):
            CypherRetriever(
                driver=self.driver,
                query="SELECT * FROM movies",  # SQL, not Cypher
                parameters=self.valid_parameters,
            )

    def test_init_undefined_parameters(self) -> None:
        # Test initialization with undefined parameters in query
        with pytest.raises(RetrieverInitializationError):
            CypherRetriever(
                driver=self.driver,
                query="MATCH (m:Movie {title: $movie_title, year: $year}) RETURN m",
                parameters=self.valid_parameters,  # Missing 'year' parameter
            )

    def test_init_invalid_parameter_type(self) -> None:
        # Test initialization with invalid parameter type
        with pytest.raises(RetrieverInitializationError):
            CypherRetriever(
                driver=self.driver,
                query=self.valid_query,
                parameters={
                    "movie_title": {
                        "type": "invalid_type",
                        "description": "Title of a movie",
                    }
                },
            )

    def test_search_success(self) -> None:
        # Test successful search
        retriever = CypherRetriever(
            driver=self.driver,
            query=self.valid_query,
            parameters=self.valid_parameters,
        )
        result = retriever.search(parameters={"movie_title": "The Matrix"})

        # Assert driver.execute_query was called with the right parameters
        self.driver.execute_query.assert_called_once()
        assert result.items
        assert result.metadata and "cypher" in result.metadata
        assert result.metadata["cypher"] == self.valid_query

    def test_search_missing_required_parameter(self) -> None:
        # Test search with missing required parameter
        retriever = CypherRetriever(
            driver=self.driver,
            query=self.valid_query,
            parameters=self.valid_parameters,
        )
        with pytest.raises(SearchValidationError):
            retriever.search(parameters={})  # Missing 'movie_title'

    def test_search_unexpected_parameter(self) -> None:
        # Test search with unexpected parameter
        retriever = CypherRetriever(
            driver=self.driver,
            query=self.valid_query,
            parameters=self.valid_parameters,
        )
        with pytest.raises(SearchValidationError):
            retriever.search(
                parameters={"movie_title": "The Matrix", "year": 1999}
            )  # 'year' not defined

    def test_search_type_mismatch(self) -> None:
        # Test search with parameter type mismatch
        retriever = CypherRetriever(
            driver=self.driver,
            query=self.valid_query,
            parameters=self.valid_parameters,
        )
        with pytest.raises(SearchValidationError):
            retriever.search(
                parameters={"movie_title": 123}
            )  # Integer, expected string

    def test_different_parameter_types(self) -> None:
        # Test with different parameter types
        query = (
            "MATCH (m:Movie) WHERE m.title = $title AND m.year = $year AND m.rating > $rating "
            "AND m.is_available = $available AND m.genres IN $genres RETURN m"
        )
        parameters = {
            "title": {"type": "string", "description": "Movie title"},
            "year": {"type": "integer", "description": "Release year"},
            "rating": {"type": "number", "description": "Minimum rating"},
            "available": {"type": "boolean", "description": "Is the movie available"},
            "genres": {"type": "array", "description": "List of genres"},
        }

        retriever = CypherRetriever(
            driver=self.driver,
            query=query,
            parameters=parameters,
        )

        # Valid parameters of different types
        result = retriever.search(
            parameters={
                "title": "The Matrix",
                "year": 1999,
                "rating": 8.5,
                "available": True,
                "genres": ["Action", "Sci-Fi"],
            }
        )

        assert result.items

        # Test integer type validation
        with pytest.raises(SearchValidationError):
            retriever.search(
                parameters={
                    "title": "The Matrix",
                    "year": "1999",  # String, expected integer
                    "rating": 8.5,
                    "available": True,
                    "genres": ["Action", "Sci-Fi"],
                }
            )

        # Test number type validation
        with pytest.raises(SearchValidationError):
            retriever.search(
                parameters={
                    "title": "The Matrix",
                    "year": 1999,
                    "rating": "8.5",  # String, expected number
                    "available": True,
                    "genres": ["Action", "Sci-Fi"],
                }
            )

        # Test boolean type validation
        with pytest.raises(SearchValidationError):
            retriever.search(
                parameters={
                    "title": "The Matrix",
                    "year": 1999,
                    "rating": 8.5,
                    "available": "yes",  # String, expected boolean
                    "genres": ["Action", "Sci-Fi"],
                }
            )

        # Test array type validation
        with pytest.raises(SearchValidationError):
            retriever.search(
                parameters={
                    "title": "The Matrix",
                    "year": 1999,
                    "rating": 8.5,
                    "available": True,
                    "genres": "Action, Sci-Fi",  # String, expected array
                }
            )

    def test_custom_result_formatter(self) -> None:
        # Test with custom result formatter
        def custom_formatter(record: Record) -> RetrieverResultItem:
            return RetrieverResultItem(
                content=f"Movie: {record['m']['title']}",
                metadata={"score": record["score"]},
            )

        retriever = CypherRetriever(
            driver=self.driver,
            query=self.valid_query,
            parameters=self.valid_parameters,
            result_formatter=custom_formatter,
        )

        result = retriever.search(parameters={"movie_title": "The Matrix"})
        assert result.items[0].content == "Movie: Test Movie"
        if result.items[0].metadata:
            assert result.items[0].metadata.get("score") == 0.9

    def test_optional_parameters(self) -> None:
        # Test with optional parameters
        query = "MATCH (m:Movie {title: $title}) WHERE m.year = $year RETURN m"
        parameters = {
            "title": {"type": "string", "description": "Movie title", "required": True},
            "year": {
                "type": "integer",
                "description": "Release year",
                "required": False,
            },
        }

        retriever = CypherRetriever(
            driver=self.driver,
            query=query,
            parameters=parameters,
        )

        # Should succeed with only required parameters
        result = retriever.search(parameters={"title": "The Matrix"})
        assert result.items

        # Should also succeed with optional parameters
        result = retriever.search(parameters={"title": "The Matrix", "year": 1999})
        assert result.items


if __name__ == "__main__":
    unittest.main()
