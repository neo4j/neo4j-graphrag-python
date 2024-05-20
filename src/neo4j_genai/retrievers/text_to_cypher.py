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
from pydantic import ValidationError

from neo4j_genai.llm import LLM
from neo4j_genai.prompts import TEXT_TO_CYPHER_PROMPT
from neo4j_genai.retrievers.base import Retriever
from neo4j_genai.schema import get_schema
from neo4j_genai.types import (
    LLMModel,
    Neo4jDriverModel,
    Neo4jSchemaModel,
    TextToCypherRetrieverModel,
    TextToCypherSearchModel,
)

logger = logging.getLogger(__name__)


# TODO: Add unit tests
class TextToCypherRetriever(Retriever):
    def __init__(
        self, driver: neo4j.Driver, llm: LLM, neo4j_schema: Optional[str] = None
    ) -> None:
        try:
            driver_model = Neo4jDriverModel(driver=driver)
            llm_model = LLMModel(llm=llm)
            neo4j_schema_model = (
                Neo4jSchemaModel(neo4j_schema=neo4j_schema) if neo4j_schema else None
            )
            validated_data = TextToCypherRetrieverModel(
                driver_model=driver_model,
                llm_model=llm_model,
                neo4j_schema_model=neo4j_schema_model,
            )
        except ValidationError as e:
            raise ValueError(f"Validation failed: {e.errors()}")

        super().__init__(validated_data.driver_model.driver)
        self.llm = validated_data.llm_model.llm
        self.neo4j_schema = (
            validated_data.neo4j_schema_model.neo4j_schema
            if validated_data.neo4j_schema_model
            else get_schema(validated_data.driver_model.driver)
        )

    def search(
        self, query_text: str, examples: Optional[list[str]] = None
    ) -> list[neo4j.Record]:
        try:
            validated_data = TextToCypherSearchModel(
                query_text=query_text, examples=examples
            )
        except ValidationError as e:
            raise ValueError(f"Validation failed: {e.errors()}")

        prompt = TEXT_TO_CYPHER_PROMPT.format(
            schema=self.neo4j_schema,
            examples="\n".join(validated_data.examples)
            if validated_data.examples
            else "",
            input=validated_data.query_text,
        )
        logger.debug("TextToCypherRetriever prompt:\n%s", prompt)

        t2c_query = self.llm.invoke(prompt)
        logger.debug("TextToCypherRetriever Cypher query: %s", t2c_query)

        records, _, _ = self.driver.execute_query(query_=t2c_query)

        return records
