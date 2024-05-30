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
import logging
from typing import Optional

import neo4j
from neo4j.exceptions import CypherSyntaxError, DriverError, Neo4jError
from pydantic import ValidationError

from neo4j_genai.llm import LLM
from neo4j_genai.prompts import TEXT2CYPHER_PROMPT
from neo4j_genai.retrievers.base import Retriever
from neo4j_genai.schema import get_schema
from neo4j_genai.types import (
    LLMModel,
    Neo4jDriverModel,
    Neo4jSchemaModel,
    Text2CypherRetrieverModel,
    Text2CypherSearchModel,
    RetrieverRawResult,
)

logger = logging.getLogger(__name__)


class Text2CypherRetriever(Retriever):
    """
    Allows for the retrieval of records from a Neo4j database using natural language.
    Converts a user's natural language query to a Cypher query using an LLM,
    then retrieves records from a Neo4j database using the generated Cypher query
    """

    def __init__(
        self, driver: neo4j.Driver, llm: LLM, neo4j_schema: Optional[str] = None
    ) -> None:
        try:
            driver_model = Neo4jDriverModel(driver=driver)
            llm_model = LLMModel(llm=llm)
            neo4j_schema_model = (
                Neo4jSchemaModel(neo4j_schema=neo4j_schema) if neo4j_schema else None
            )
            validated_data = Text2CypherRetrieverModel(
                driver_model=driver_model,
                llm_model=llm_model,
                neo4j_schema_model=neo4j_schema_model,
            )
        except ValidationError as e:
            raise ValueError(f"Validation failed: {e.errors()}")

        super().__init__(validated_data.driver_model.driver)
        self.llm = validated_data.llm_model.llm
        try:
            self.neo4j_schema = (
                validated_data.neo4j_schema_model.neo4j_schema
                if validated_data.neo4j_schema_model
                else get_schema(validated_data.driver_model.driver)
            )
        except Neo4jError as e:
            logger.error("Failed to get schema for Text2CypherRetriever: %s", e.message)
            raise RuntimeError(
                f"Failed to get schema for Text2CypherRetriever: {e.message}"
            )
        except DriverError as e:
            logger.error("Failed to get schema for Text2CypherRetriever:")
            logger.exception(e)
            raise e

    def get_search_results(
        self, query_text: str, examples: Optional[list[str]] = None
    ) -> RetrieverRawResult:
        """Converts query_text to a Cypher query using an LLM.
           Retrieve records from a Neo4j database using the generated Cypher query.

        Args:
            query_text (str): The natural language query used to search the Neo4j database.
            examples (Optional[list[str], optional): Optional user input/query pairs for the LLM to use as examples.

        Raises:
            ValidationError: If validation of the input arguments fail.
            RuntimeError: If the LLM fails to generate a correct Cypher query.

        Returns:
            list[neo4j.Record]: The results of the search query.
        """
        try:
            validated_data = Text2CypherSearchModel(
                query_text=query_text, examples=examples
            )
        except ValidationError as e:
            raise ValueError(f"Validation failed: {e.errors()}")

        prompt = TEXT2CYPHER_PROMPT.format(
            schema=self.neo4j_schema,
            examples="\n".join(validated_data.examples)
            if validated_data.examples
            else "",
            input=validated_data.query_text,
        )
        logger.debug("Text2CypherRetriever prompt: %s", prompt)

        try:
            t2c_query = self.llm.invoke(prompt)
            logger.debug("Text2CypherRetriever Cypher query: %s", t2c_query)
            records, _, _ = self.driver.execute_query(query_=t2c_query)
        except CypherSyntaxError as e:
            logger.error("Cypher query generation failed: %s", e.message)
            raise RuntimeError(f"Cypher query generation failed: {e.message}")

        return RetrieverRawResult(
            records=records,
            metadata={
                "cypher": t2c_query,
            },
        )
