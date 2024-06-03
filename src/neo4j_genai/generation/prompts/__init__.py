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

    def format(self, *args, **kwargs):
        for e in self.EXPECTED_INPUTS:
            if e not in kwargs:
                raise Exception(f"Missing input {e}")
        return self.template.format(**kwargs)


class RagTemplate(PromptTemplate):
    DEFAULT_TEMPLATE = """Answer the user question using the following context

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    EXPECTED_INPUTS = ["context", "query"]

    def format(self, query: str, context: str) -> str:
        return super().format(query=query, context=context)


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
