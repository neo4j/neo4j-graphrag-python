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

from typing import Any, Optional

from neo4j_genai.exceptions import PromptMissingInputError


class PromptTemplate:
    """This class is used to generate a parameterized prompt. It is defined
    from a string (the template) using the Python format syntax (parameters
    between curly braces `{}`) and a list of required inputs.
    Before sending the instructions to an LLM, call the `format` method that will
    replace parameters with the provided values. If any of the expected inputs is
    missing, a `PromptMissingInputError` is raised.
    """

    DEFAULT_TEMPLATE: str = ""
    EXPECTED_INPUTS: list[str] = []

    def __init__(
        self,
        template: Optional[str] = None,
        expected_inputs: Optional[list[str]] = None,
    ) -> None:
        self.template = template or self.DEFAULT_TEMPLATE
        self.expected_inputs = expected_inputs or self.EXPECTED_INPUTS

    def _format(self, **kwargs: Any) -> str:
        for e in self.EXPECTED_INPUTS:
            if e not in kwargs:
                raise PromptMissingInputError(f"Missing input '{e}'")
        return self.template.format(**kwargs)

    def format(self, *args: Any, **kwargs: Any) -> str:
        """This method is used to replace parameters with the provided values.
        Parameters must be provided:
        - as kwargs
        - as args if using the same order as in the expected inputs

        Example:

        .. code-block:: python

            prompt_template = PromptTemplate(
                template='''Explain the following concept to {target_audience}:
                Concept: {concept}
                Answer:
                ''',
                expected_inputs=['target_audience', 'concept']
            )
            prompt = prompt_template.format('12 yo children', concept='graph database')
            print(prompt)

            # Result:
            # '''Explain the following concept to 12 yo children:
            # Concept: graph database
            # Answer:
            # '''

        """
        data = kwargs
        data.update({k: v for k, v in zip(self.expected_inputs, args)})
        return self._format(**data)


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
