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

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union

import neo4j
from neo4j.exceptions import CypherSyntaxError
from pydantic import ValidationError

from neo4j_graphrag.exceptions import RetrieverInitializationError, SearchValidationError
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import (
    CypherParameterDefinition,
    CypherParameterType,
    CypherRetrieverModel,
    CypherSearchModel,
    Neo4jDriverModel,
    RawSearchResult,
    RetrieverResultItem,
)

logger = logging.getLogger(__name__)


class CypherRetriever(Retriever):
    """
    Allows for the retrieval of records from a Neo4j database using a parameterized Cypher query.
    
    This retriever enables direct execution of predefined Cypher queries with dynamic parameters.
    It ensures type safety through parameter validation and provides the standard retriever result format.

    Example:
    
    .. code-block:: python

        import neo4j
        from neo4j_graphrag.retrievers import CypherRetriever

        driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)

        # Create a retriever for finding movies by title
        retriever = CypherRetriever(
            driver=driver,
            query="MATCH (m:Movie {title: $movie_title}) RETURN m",
            parameters={
                "movie_title": {
                    "type": "string", 
                    "description": "Title of a movie"
                }
            }
        )

        # Use the retriever with specific parameter values
        results = retriever.search(parameters={"movie_title": "The Matrix"})

    Args:
        driver (neo4j.Driver): The Neo4j Python driver.
        query (str): Cypher query with parameter placeholders.
        parameters (Dict[str, Dict]): Parameter definitions with types and descriptions.
            Each parameter should have a 'type' and 'description' field.
            Supported types: 'string', 'number', 'integer', 'boolean', 'array'.
        result_formatter (Optional[Callable[[neo4j.Record], RetrieverResultItem]]): 
            Custom function to transform a neo4j.Record to a RetrieverResultItem.
        neo4j_database (Optional[str]): The name of the Neo4j database to use.

    Raises:
        RetrieverInitializationError: If validation of the input arguments fail.
    """

    def __init__(
        self,
        driver: neo4j.Driver,
        query: str,
        parameters: Dict[str, Dict],
        result_formatter: Optional[Callable[[neo4j.Record], RetrieverResultItem]] = None,
        neo4j_database: Optional[str] = None,
    ) -> None:
        # Convert parameter dictionaries to CypherParameterDefinition objects
        param_definitions = {}
        for param_name, param_def in parameters.items():
            param_type = param_def.get("type", "string")
            description = param_def.get("description", "")
            required = param_def.get("required", True)
            
            try:
                param_definitions[param_name] = CypherParameterDefinition(
                    type=param_type,
                    description=description,
                    required=required
                )
            except ValidationError as e:
                raise RetrieverInitializationError(
                    f"Invalid parameter definition for {param_name}: {e.errors()}"
                ) from e

        try:
            driver_model = Neo4jDriverModel(driver=driver)
            validated_data = CypherRetrieverModel(
                driver_model=driver_model,
                query=query,
                parameters=param_definitions,
                result_formatter=result_formatter,
                neo4j_database=neo4j_database,
            )
        except ValidationError as e:
            raise RetrieverInitializationError(e.errors()) from e

        # Validate that the query is syntactically valid Cypher
        self._validate_cypher_query(query)
        
        # Validate that all parameters in the query are defined
        self._validate_query_parameters(query, param_definitions)

        super().__init__(validated_data.driver_model.driver, validated_data.neo4j_database)
        self.query = validated_data.query
        self.parameters = validated_data.parameters
        self.result_formatter = validated_data.result_formatter
    
    def _validate_cypher_query(self, query: str) -> None:
        """
        Validates that the query is syntactically valid Cypher.
        
        Args:
            query (str): The Cypher query to validate.
            
        Raises:
            RetrieverInitializationError: If the query is not valid Cypher.
        """
        # We can't fully validate the query without executing it, but we can check for basic syntax
        if not query.strip():
            raise RetrieverInitializationError("Query cannot be empty")
        
        # Check for presence of common Cypher keywords
        if not any(keyword in query.upper() for keyword in ["MATCH", "RETURN", "CREATE", "MERGE", "WITH"]):
            raise RetrieverInitializationError(
                "Query does not appear to be valid Cypher. "
                "It should contain at least one of: MATCH, RETURN, CREATE, MERGE, WITH"
            )

    def _validate_query_parameters(self, query: str, parameters: Dict[str, CypherParameterDefinition]) -> None:
        """
        Validates that all parameters in the query are defined in the parameters dictionary.
        
        Args:
            query (str): The Cypher query to validate.
            parameters (Dict[str, CypherParameterDefinition]): The parameter definitions.
            
        Raises:
            RetrieverInitializationError: If any parameters in the query are not defined.
        """
        # Find all parameters in the query (starting with $)
        param_pattern = r'\$([a-zA-Z0-9_]+)'
        query_params = set(re.findall(param_pattern, query))
        
        # Check that all parameters in the query are defined
        undefined_params = query_params - set(parameters.keys())
        if undefined_params:
            raise RetrieverInitializationError(
                f"The following parameters are used in the query but not defined: {', '.join(undefined_params)}"
            )

    def _validate_parameter_values(self, parameters: Dict[str, Any]) -> None:
        """
        Validates that parameter values match their defined types.
        
        Args:
            parameters (Dict[str, Any]): The parameter values to validate.
            
        Raises:
            SearchValidationError: If any parameter values do not match their defined types.
        """
        # Check that all required parameters are provided
        for param_name, param_def in self.parameters.items():
            if param_def.required and param_name not in parameters:
                raise SearchValidationError(f"Required parameter '{param_name}' is missing")

        # Validate the type of each parameter
        for param_name, param_value in parameters.items():
            if param_name not in self.parameters:
                raise SearchValidationError(f"Unexpected parameter: {param_name}")
            
            param_def = self.parameters[param_name]
            
            # Type validation
            if param_def.type == CypherParameterType.STRING:
                if not isinstance(param_value, str):
                    raise SearchValidationError(
                        f"Parameter '{param_name}' should be of type string, got {type(param_value).__name__}"
                    )
            elif param_def.type == CypherParameterType.NUMBER:
                if not isinstance(param_value, (int, float)):
                    raise SearchValidationError(
                        f"Parameter '{param_name}' should be of type number, got {type(param_value).__name__}"
                    )
            elif param_def.type == CypherParameterType.INTEGER:
                if not isinstance(param_value, int) or isinstance(param_value, bool):
                    raise SearchValidationError(
                        f"Parameter '{param_name}' should be of type integer, got {type(param_value).__name__}"
                    )
            elif param_def.type == CypherParameterType.BOOLEAN:
                if not isinstance(param_value, bool):
                    raise SearchValidationError(
                        f"Parameter '{param_name}' should be of type boolean, got {type(param_value).__name__}"
                    )
            elif param_def.type == CypherParameterType.ARRAY:
                if not isinstance(param_value, (list, tuple)):
                    raise SearchValidationError(
                        f"Parameter '{param_name}' should be of type array, got {type(param_value).__name__}"
                    )

    def get_search_results(self, parameters: Dict[str, Any]) -> RawSearchResult:
        """
        Executes the Cypher query with the provided parameters and returns the results.
        
        Args:
            parameters (Dict[str, Any]): Parameter values to use in the query.
                Each parameter should match the type specified in the parameter definitions.
        
        Raises:
            SearchValidationError: If validation of the parameters fails.
        
        Returns:
            RawSearchResult: The results of the query as a list of neo4j.Record and an optional metadata dict.
        """
        try:
            validated_data = CypherSearchModel(parameters=parameters)
        except ValidationError as e:
            raise SearchValidationError(e.errors()) from e
        
        # Validate parameter values against their definitions
        self._validate_parameter_values(validated_data.parameters)
        
        logger.debug("CypherRetriever query: %s", self.query)
        logger.debug("CypherRetriever parameters: %s", validated_data.parameters)
        
        try:
            records, _, _ = self.driver.execute_query(
                query_=self.query,
                parameters_=validated_data.parameters,
                database_=self.neo4j_database,
                routing_=neo4j.RoutingControl.READ,
            )
        except CypherSyntaxError as e:
            raise SearchValidationError(f"Cypher syntax error: {e.message}") from e
        except Exception as e:
            raise SearchValidationError(f"Failed to execute query: {str(e)}") from e
        
        return RawSearchResult(
            records=records,
            metadata={
                "cypher": self.query,
            },
        )