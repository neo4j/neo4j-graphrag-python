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


class RetrieverInitializationError(Exception):
    """Exception raised when initialization of a retriever fails."""

    def __init__(self, errors: str):
        super().__init__(f"Initialization failed: {errors}")
        self.errors = errors


class SearchValidationError(Exception):
    """Exception raised for validation errors during search."""

    def __init__(self, errors):
        super().__init__(f"Search validation failed: {errors}")
        self.errors = errors


class FilterValidationError(Exception):
    """Exception raised when input validation for metadata filtering fails."""

    pass


class EmbeddingRequiredError(Exception):
    """Exception raised when an embedding method is required but not provided."""

    pass


class RecordCreationError(Exception):
    """Exception raised when valid Record fails to be created."""

    pass


class Neo4jIndexError(Exception):
    """Exception raised when handling Neo4j indexes fails."""

    pass


class Neo4jVersionError(Exception):
    """Exception raised when Neo4j version does not meet minimum requirements."""

    def __init__(self):
        super().__init__("This package only supports Neo4j version 5.18.1 or greater")
