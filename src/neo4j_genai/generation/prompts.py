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

from jinja2 import Environment, StrictUndefined


class PromptTemplate:
    """This class is used to generate a parameterized prompt. It is defined
    from a string (the template) using the Jinja2 templating system syntax
    (parameters between double curly braces `{{ }}`) .
    Before sending the instructions to an LLM, call the `format` method that will
    render the template, replacing parameters with the provided values.
    """

    DEFAULT_TEMPLATE: str = ""

    def __init__(
        self,
        template: Optional[str] = None,
    ) -> None:
        self._string_template = template or self.DEFAULT_TEMPLATE
        env = Environment(undefined=StrictUndefined)
        self._template = env.from_string(self._string_template)

    @property
    def _local_context(self) -> dict[str, Any]:
        return {}

    def render(self, **kwargs: Any) -> str:
        """This method is used to replace parameters with the provided values.

        Example:

        .. code-block:: python

            prompt_template = PromptTemplate(
                template='''Explain the following concept to {{target_audience}}:
                Concept: {{concept}}
                Answer:
                ''',
            )
            prompt = prompt_template.format(target_audience='12 yo children', concept='graph database')
            print(prompt)

            # Result:
            # '''Explain the following concept to 12 yo children:
            # Concept: graph database
            # Answer:
            # '''

        """
        local_context = self._local_context
        return self._template.render(local_context=local_context, **kwargs)  # type: ignore

    def format(self, *args: Any, **kwargs: Any) -> str:
        return self.render(**kwargs)


class RagTemplate(PromptTemplate):
    DEFAULT_TEMPLATE = """Answer the user question using the following context

Context:
{{context}}

Examples:
{{examples}}

Question:
{{query}}

Answer:
"""

    def format(self, query: str, context: str, examples: str) -> str:
        return self.render(query=query, context=context, examples=examples)


class Text2CypherTemplate(PromptTemplate):
    DEFAULT_TEMPLATE = """
Task: Generate a Cypher statement for querying a Neo4j graph database from a user input.

Schema:
{{schema}}

Examples (optional):
{{examples}}

Input:
{{query}}

Do not use any properties or relationships not included in the schema.
Do not include triple backticks ``` or any additional text except the generated Cypher statement in your response.

Cypher query:
"""

    def format(self, query: str, schema: str, examples: str) -> str:
        return self.render(query=query, schema=schema, examples=examples)


class ERExtractionTemplate(PromptTemplate):
    DEFAULT_TEMPLATE = """
{% macro property_list(properties) %}
    {%- for prop in properties %}
    - {{ prop.name }} ({{prop.type}}) {% endfor %}
{% endmacro %}

You are a top-tier algorithm designed for extracting
information in structured formats to build a knowledge graph.

Extract the entities and specify their type from the following text.
Also extract the relations between these entities.

Return result as JSON using the following format:
{"entities": [{"id": 0, "label": "", "properties": [{"name": "", "value": ""}]},],
"relations": [{"from": 0, "to": 1, "properties": [{"name": "", "value": ""}]}, ]}

Use only fhe following entities and relations:
Entities:
{% for entity in schema.entities %}
- {{ entity.label }}:
{{ property_list(entity.properties) }}
{% endfor %}

Relations:
{% for rel in schema.relations %}
- {{ rel.label }}:
    - from {{ rel.source_node_type }}
    - to {{ rel.target_node_type }}
{{ property_list(rel.properties) }}
{%- endfor -%}

Assign a unique ID to each entity, and reuse it to define relationships.
Do respect the source and target entity types for relationship and
the relationship direction.

{% if examples %}
Examples
{{ examples }}
{% endif %}

Input text:

{{ text }}
"""

    def format(self, text: str, schema: Any, examples: str) -> str:
        return self.render(text=text, schema=schema, examples=examples)
