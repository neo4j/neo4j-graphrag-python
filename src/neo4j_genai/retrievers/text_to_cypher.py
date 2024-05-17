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
import neo4j
from neo4j_genai.retrievers.base import Retriever
from neo4j_genai.llm import LLM
from neo4j_genai.prompts import TEXT_TO_CYPHER_PROMPT
from typing import Optional
import logging


logger = logging.getLogger(__name__)

# TODO: Add Pydantic input validation like other retrievers
# TODO: Add unit tests
class TextToCypherRetriever(Retriever):

    def __init__(self, driver: neo4j.Driver, llm: LLM, schema: str):
        super().__init__(driver)
        self.llm = llm
        # TODO: Decide if the schema should be provided as an input when creating an instance of the class
        #       or should be retrieved by the class itself
        self.schema = schema

    def search(self, query_text: str, examples: Optional[list[str]] = None) -> list[neo4j.Record]:

        # TODO: Decide if we want a full prompt template class similar to LangChain or if
        #       string formatting is enough
        prompt = TEXT_TO_CYPHER_PROMPT.format(
            schema=self.schema, examples="\n".join(examples) if examples else "", input=query_text
        )
        logger.debug("TextToCypherRetriever prompt:\n%s", t2c_query)

        # TODO: Fail here if the LLM doesn't generate a valid Cypher query
        t2c_query = self.llm.invoke(prompt)
        logger.debug("TextToCypherRetriever Cypher query: %s", t2c_query)

        # TODO: Fail here if the query isn't valid for the specific database
        records, _, _ = self.driver.execute_query(query_=t2c_query)

        return records
