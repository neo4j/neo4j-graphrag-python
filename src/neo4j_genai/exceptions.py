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
from pydantic_core import ErrorDetails


class Neo4jGenAiError(Exception):
    """Global exception used for the neo4j-genai package."""

    pass


class RetrieverInitializationError(Neo4jGenAiError):
    """Exception raised when initialization of a retriever fails."""

    def __init__(self, errors: list[ErrorDetails]) -> None:
        super().__init__(f"Initialization failed: {errors}")
        self.errors = errors


class SearchValidationError(Neo4jGenAiError):
    """Exception raised for validation errors during search."""

    def __init__(self, errors: list[ErrorDetails]) -> None:
        super().__init__(f"Search validation failed: {errors}")
        self.errors = errors


class FilterValidationError(Neo4jGenAiError):
    """Exception raised when input validation for metadata filtering fails."""

    pass


class EmbeddingRequiredError(Neo4jGenAiError):
    """Exception raised when an embedding method is required but not provided."""

    pass


class InvalidRetrieverResultError(Neo4jGenAiError):
    """Exception raised when the Retriever fails to return a result."""

    pass


class Neo4jIndexError(Neo4jGenAiError):
    """Exception raised when handling Neo4j index fails."""

    pass


class Neo4jVersionError(Neo4jGenAiError):
    """Exception raised when Neo4j version does not meet minimum requirements."""

    def __init__(self) -> None:
        super().__init__("This package only supports Neo4j version 5.18.1 or greater")


class Text2CypherRetrievalError(Neo4jGenAiError):
    """Exception raised when text-to-cypher retrieval fails."""

    pass


class SchemaFetchError(Neo4jGenAiError):
    """Exception raised when a Neo4jSchema cannot be fetched."""

    pass
