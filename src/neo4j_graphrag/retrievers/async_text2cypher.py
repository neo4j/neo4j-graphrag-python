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
from typing import Any, Callable, Dict, Optional

import neo4j
from neo4j.exceptions import CypherSyntaxError, DriverError, Neo4jError
from pydantic import ValidationError

from neo4j_graphrag.exceptions import (
    RetrieverInitializationError,
    SchemaFetchError,
    SearchValidationError,
    Text2CypherRetrievalError,
)
from neo4j_graphrag.generation.prompts import Text2CypherTemplate
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.retrievers.base import AsyncRetriever
from neo4j_graphrag.retrievers.text2cypher import extract_cypher, READ_ONLY_QUERY_TYPE
from neo4j_graphrag.schema import get_schema
from neo4j_graphrag.types import (
    AsyncNeo4jDriverModel,
    AsyncText2CypherRetrieverModel,
    LLMModel,
    Neo4jSchemaModel,
    RawSearchResult,
    RetrieverResultItem,
    Text2CypherSearchModel,
)

logger = logging.getLogger(__name__)


class AsyncText2CypherRetriever(AsyncRetriever):
    """
    Allows async retrieval of records from a Neo4j database using natural language.
    Converts a user's natural language query to a Cypher query using an LLM,
    then retrieves records from a Neo4j database using the generated Cypher query.

    Args:
        driver (neo4j.AsyncDriver): The Neo4j async Python driver.
        llm (LLMInterface): LLM object to generate the Cypher query.
        neo4j_schema (Optional[str]): Neo4j schema used to generate the Cypher query.
        examples (Optional[list[str]]): Optional user input/query pairs for the LLM to use as examples.
        result_formatter (Optional[Callable[[neo4j.Record], RetrieverResultItem]]): Custom formatter.
        custom_prompt (Optional[str]): Optional custom prompt template.
        neo4j_database (Optional[str]): The name of the Neo4j database.

    Raises:
        RetrieverInitializationError: If validation of the input arguments fail.
    """

    def __init__(
        self,
        driver: neo4j.AsyncDriver,
        llm: LLMInterface,
        neo4j_schema: Optional[str] = None,
        examples: Optional[list[str]] = None,
        result_formatter: Optional[
            Callable[[neo4j.Record], RetrieverResultItem]
        ] = None,
        custom_prompt: Optional[str] = None,
        neo4j_database: Optional[str] = None,
    ) -> None:
        try:
            driver_model = AsyncNeo4jDriverModel(driver=driver)
            llm_model = LLMModel(llm=llm)
            neo4j_schema_model = (
                Neo4jSchemaModel(neo4j_schema=neo4j_schema) if neo4j_schema else None
            )
            validated_data = AsyncText2CypherRetrieverModel(
                driver_model=driver_model,
                llm_model=llm_model,
                neo4j_schema_model=neo4j_schema_model,
                examples=examples,
                result_formatter=result_formatter,
                custom_prompt=custom_prompt,
                neo4j_database=neo4j_database,
            )
        except ValidationError as e:
            raise RetrieverInitializationError(e.errors()) from e

        super().__init__(
            validated_data.driver_model.driver, validated_data.neo4j_database
        )
        self.llm = validated_data.llm_model.llm
        self.examples = validated_data.examples
        self.result_formatter = validated_data.result_formatter
        self.custom_prompt = validated_data.custom_prompt
        self.neo4j_schema = neo4j_schema or ""

    async def async_init(self) -> "AsyncText2CypherRetriever":
        """Fetch Neo4j schema if not provided. Must be awaited after construction."""
        if not self.neo4j_schema:
            try:
                self.neo4j_schema = get_schema(self.driver)
            except (Neo4jError, DriverError) as e:
                error_message = getattr(e, "message", str(e))
                raise SchemaFetchError(
                    f"Failed to fetch schema for AsyncText2CypherRetriever: {error_message}"
                ) from e
        return self

    async def get_search_results(
        self,
        query_text: str,
        prompt_params: Optional[Dict[str, Any]] = None,
    ) -> RawSearchResult:
        """Async version of Text2CypherRetriever.get_search_results.

        Converts query_text to a Cypher query using an LLM, then retrieves
        records from Neo4j using the generated Cypher query.

        Args:
            query_text (str): The natural language query.
            prompt_params (Optional[Dict[str, Any]]): Additional prompt parameters.

        Raises:
            SearchValidationError: If validation of the input arguments fail.
            Text2CypherRetrievalError: If the LLM fails to generate a correct Cypher query.

        Returns:
            RawSearchResult: The results of the search query.
        """
        try:
            validated_data = Text2CypherSearchModel(query_text=query_text)
        except ValidationError as e:
            raise SearchValidationError(e.errors()) from e

        prompt_template = Text2CypherTemplate(template=self.custom_prompt)

        if prompt_params is not None:
            examples_to_use = prompt_params.pop("examples", None) or (
                "\n".join(self.examples) if self.examples else ""
            )
            schema_to_use = prompt_params.pop("schema", None) or self.neo4j_schema
        else:
            examples_to_use = "\n".join(self.examples) if self.examples else ""
            schema_to_use = self.neo4j_schema
            prompt_params = dict()

        prompt = prompt_template.format(
            schema=schema_to_use,
            examples=examples_to_use,
            query_text=validated_data.query_text,
            **prompt_params,
        )

        logger.debug("AsyncText2CypherRetriever prompt: %s", prompt)

        try:
            llm_result = self.llm.invoke(prompt)
            t2c_query = extract_cypher(llm_result.content)
            logger.debug("AsyncText2CypherRetriever Cypher query: %s", t2c_query)

            explain_result = await self.driver.execute_query(
                query_=f"EXPLAIN {t2c_query}",
                database_=self.neo4j_database,
                routing_=neo4j.RoutingControl.READ,
            )
            explain_summary = explain_result.summary
            if explain_summary.query_type != READ_ONLY_QUERY_TYPE:
                raise Text2CypherRetrievalError(
                    "Refusing to execute non-read-only Cypher "
                    f"(query_type={explain_summary.query_type!r}): {t2c_query}"
                )
            result = await self.driver.execute_query(
                query_=t2c_query,
                database_=self.neo4j_database,
                routing_=neo4j.RoutingControl.READ,
            )
            records = result.records
        except CypherSyntaxError as e:
            raise Text2CypherRetrievalError(
                f"Failed to get search result: {e.message}"
            ) from e

        return RawSearchResult(
            records=records,
            metadata={"cypher": t2c_query},
        )
