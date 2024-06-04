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

from typing import Optional


class PromptTemplate:
    DEFAULT_TEMPLATE: str = ""
    EXPECTED_INPUTS: Optional[list] = None

    def __init__(
        self,
        template: Optional[str] = None,
        expected_inputs: Optional[list[str]] = None,
    ):
        self.template = template or self.DEFAULT_TEMPLATE
        self.expected_inputs = expected_inputs or self.EXPECTED_INPUTS or []

    def format(self, **kwargs):
        for e in self.EXPECTED_INPUTS:
            if e not in kwargs:
                raise Exception(f"Missing input {e}")
        return self.template.format(**kwargs)


class RagTemplate(PromptTemplate):
    DEFAULT_TEMPLATE = """Answer the user question using the following context

Context:
{context}

Examples:
{examples}

Question:
{query}

Answer:
"""
    EXPECTED_INPUTS = ["context", "query", "examples"]

    def format(self, query: str, context: str, examples: str) -> str:
        return super().format(query=query, context=context, examples=examples)


class Text2CypherTemplate(PromptTemplate):
    DEFAULT_TEMPLATE = """
Task: Generate a Cypher statement for querying a Neo4j graph database from a user input.

Schema:
{schema}

Examples (optional):
{examples}

Input:
{query}

Do not use any properties or relationships not included in the schema.
Do not include triple backticks ``` or any additional text except the generated Cypher statement in your response.

Cypher query:
"""
    EXPECTED_INPUTS = ["schema", "query", "examples"]

    def format(self, query: str, schema: str, examples: str) -> str:
        return super().format(query=query, schema=schema, examples=examples)
